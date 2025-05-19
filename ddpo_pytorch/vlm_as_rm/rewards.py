from PIL import Image
import io
import numpy as np
import torch
import requests
import json
import base64

def evaluate_image_with_prompt():
    def get_image_base64(image):
        """
        Accepts an image (NumPy array or PyTorch tensor), processes it, and 
        returns the Base64-encoded data URL string.
        """

        # Ensure the image is in the correct format and data type
        if isinstance(image, torch.Tensor):
            # If the input is a torch.Tensor, convert to uint8 format
            image = (image * 255).round().clamp(0, 255).to(torch.uint8)
        elif isinstance(image, np.ndarray):
            # If the input is a NumPy array, ensure it's in NHWC format
            if image.shape[0] == 3:  # Assume CHW format
                image = image.transpose(1, 2, 0)  # Convert to HWC format
            image = torch.tensor(image, dtype=torch.uint8)
        else:
            raise TypeError("Unsupported image type. Must be a NumPy array or PyTorch tensor.")

        # Encode the image to Base64
        image = image.permute(1, 2, 0).cpu().numpy()
        # Convert the image to a PIL Image
        pil_image = Image.fromarray(image)
        # Save the image to a bytes buffer in JPEG format
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        # Encode the image bytes to Base64
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        # Return the Base64-encoded data URL
        return f"data:image/jpeg;base64,{image_base64}"
    
    def _fn(image, prompt, metadata=None):
        print(prompt)
        # Initialize the OpenAI client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",  # Replace with your OpenRouter API key
        )
        # Path to the flower image
        base64_image_url = get_image_base64(image)

        # Create a chat completion request
        completion = client.chat.completions.create(
            model="qwen/qwen2.5-vl-3b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are a helpful assistant who can evaluate images against specific prompts. Please determine if the image satisfies the following key point: {prompt}.\
                                    First thinking and tell me how the image related to the key point,i.e. identify which part of picture related which description of image. Then add a newline\
                                    character after the think part, then answer in one of the following categories: 'Yes', 'Weak Yes', 'Weak No', 'No'. With format like: Answer: <Yes>"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image_url
                            }
                        }
                    ]
                }
            ]
        )
        res = completion.choices[0].message.content
        return res

    return _fn

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def ocr_score():
    from ddpo_pytorch.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        prompts = [prompt.split("'")[1] for prompt in prompts]
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
