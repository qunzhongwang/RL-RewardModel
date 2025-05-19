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
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def ddim_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
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
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1).to(torch.float32)
    sigma_next = self.sigmas[next_step_index].view(-1, 1, 1, 1).to(torch.float32)
    sigma = torch.clamp(sigma, min=0.001, max=0.999)
    sigma_next = torch.clamp(sigma_next, min=0.001, max=0.999)
    # Upcast to avoid precision issues when computing prev_sample
    old_dtype = model_output.dtype
    sample = sample.to(torch.float32)
    model_output = model_output.to(torch.float32)

    # prev_sample = sample + (sigma_next - sigma) * model_output
    dt = sigma_next - sigma
    # std_dev_t = torch.sqrt(2*sigma/(1-sigma))

    std_dev_t = torch.tensor(0.01, dtype=sigma.dtype, device=sigma.device)
    # std_dev_t = sigma
    # std_dev_t = torch.linspace(0.05, 0.002, steps=len(self.sigmas), dtype=sigma.dtype, device=sigma.device)[step_index].view(-1,1,1,1)
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    # prev_sample_mean = sample + dt * model_output

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
    prev_sample = prev_sample.to(torch.float32) # 防止train的时候传入的prev_sample是fp16，导致溢出
    # log prob of prev_sample given prev_sample_mean and std_dev_t
    # log_prob = (
    #     -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
    #     - torch.log(std_dev_t * torch.sqrt(-1*dt))
    #     - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    # )
    log_prob1 = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
    log_prob2 = - torch.log(std_dev_t * torch.sqrt(-1*dt))
    log_prob2 = _left_broadcast(log_prob2, log_prob1.shape)
    log_prob3 = - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))).to(sigma.device)
    log_prob3 = _left_broadcast(log_prob3, log_prob1.shape)
    log_prob = log_prob1 + log_prob2 + log_prob3

    log_prob1 = log_prob1.mean(dim=tuple(range(1, log_prob1.ndim)))
    log_prob2 = log_prob2.mean(dim=tuple(range(1, log_prob2.ndim)))
    log_prob3 = log_prob3.mean(dim=tuple(range(1, log_prob3.ndim)))
    # print('1', (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))).mean())
    # print('2', (- torch.log(std_dev_t * torch.sqrt(-1*dt))).mean())
    # print('3', (- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))).mean())
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    # breakpoint()
    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(old_dtype)
    
    return prev_sample, log_prob, log_prob1, log_prob2, log_prob3