# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
# with the following modifications:
# - It computes and returns the log prob of `prev_sample` given the Transformer prediction.
#   it uses it to compute the log prob.
# - Timesteps can be a batched torch.Tensor.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler

def ddim_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        s_churn (`float`):
        s_tmin  (`float`):
        s_tmax  (`float`):
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

    Returns:
        [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`,
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
            otherwise a tuple is returned where the first element is the sample tensor.
    """

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )
    step_index = [self.index_for_timestep(t) for t in timestep]
    next_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1)
    sigma_next = self.sigmas[next_step_index].view(-1, 1, 1, 1)
    sigma = torch.max(sigma, torch.tensor(1e-3))
    sigma_next = torch.max(sigma_next, torch.tensor(1e-3))
    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    prev_sample = sample + (sigma_next - sigma) * model_output

    sample_0 = sample - sigma * model_output
    prev_sample_mean = (1-sigma_next)*sample_0
    std_dev_t = sigma_next
    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)
    
    return prev_sample, log_prob