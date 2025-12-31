"""
rCM (Rectified Consistency Model) Sampler for TurboDiffusion

Extracted from TurboDiffusion wan2.2_i2v_infer.py
"""

import math
import torch
from tqdm import tqdm


def rcm_sampler(
    model,
    init_noise,
    condition,
    num_steps=4,
    sigma_max=200.0,
    use_ode=False,
    generator=None,
    start_step=0,
    end_step=None,
    verbose=True
):
    """
    rCM sampler for TurboDiffusion models.

    Args:
        model: TurboDiffusion WanModel
        init_noise: Initial noise tensor (B, C, T, H, W)
        condition: Conditioning dict with crossattn_emb and y_B_C_T_H_W
        num_steps: Total number of sampling steps (1-4)
        sigma_max: Initial sigma for TrigFlow
        use_ode: Whether to use ODE sampling (sharper but less robust)
        generator: Random generator for SDE sampling
        start_step: Starting step (for dual-expert switching)
        end_step: Ending step (for dual-expert switching)
        verbose: Whether to show progress bar

    Returns:
        Sampled latents (B, C, T, H, W)
    """
    device = init_noise.device
    dtype = init_noise.dtype

    if end_step is None:
        end_step = num_steps

    # TrigFlow timesteps
    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Initialize x or use provided latent
    if start_step == 0:
        x = init_noise.to(torch.float64) * t_steps[0]
    else:
        x = init_noise.to(torch.float64)

    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    # Get the relevant timestep pairs for this sampling range
    step_pairs = list(zip(t_steps[start_step:-1], t_steps[start_step+1:]))
    if end_step < num_steps:
        step_pairs = step_pairs[:end_step-start_step]

    # Sampling loop
    iterator = tqdm(step_pairs, desc=f"Sampling steps {start_step}-{end_step}", disable=not verbose)
    for t_cur, t_next in iterator:
        with torch.no_grad():
            # Model prediction
            v_pred = model(
                x_B_C_T_H_W=x.to(dtype=dtype),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(dtype=dtype),
                **condition
            ).to(torch.float64)

            # Update step
            if use_ode:
                # ODE sampling (deterministic)
                x = x - (t_cur - t_next) * v_pred
            else:
                # SDE sampling (stochastic)
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                ).to(torch.float64)

    return x.float()


# Aliases for compatibility
cm_sampler_diffusers = rcm_sampler
cm_sampler_heun = rcm_sampler
