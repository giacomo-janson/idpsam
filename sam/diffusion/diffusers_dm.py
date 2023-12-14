import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Optional, Callable

from diffusers import DDPMScheduler, DDIMScheduler
from sam.diffusion.common import DiffusionCommon


class Diffusers(DiffusionCommon):

    def __init__(self,
                 eps_model: Callable,
                 sched_params: dict,
                 loss="l2",
                 ema=None,
                 sc_params=None):

        # Setup the network and other core parameters.
        self.eps_model = eps_model
        self.ema = ema
        if not loss in ("l2",):  # ("l1", "l2", "huber"):
            raise KeyError(loss)
        self.loss_type = loss

        # Setup the diffusers scheduler.
        if sched_params["name"] == "ddpm":
            # From: https://huggingface.co/docs/diffusers/api/schedulers/ddpm
            #       https://huggingface.co/docs/diffusers/api/pipelines/ddpm
            self.sched = DDPMScheduler(
                num_train_timesteps=sched_params["num_train_timesteps"],
                beta_start=sched_params["beta_start"],
                beta_end=sched_params["beta_end"],
                beta_schedule=sched_params["beta_schedule"],
                trained_betas=None,
                variance_type=sched_params["variance_type"],
                clip_sample=False,
                clip_sample_range=1.0,
                prediction_type=sched_params["prediction_type"],
                thresholding=False,
                dynamic_thresholding_ratio=0.995,
                sample_max_value=1.0)
        elif sched_params["name"] == "ddim":
            # From: https://huggingface.co/docs/diffusers/api/schedulers/ddim
            #       https://huggingface.co/docs/diffusers/api/pipelines/ddim
            self.sched = DDIMScheduler(
                num_train_timesteps=sched_params["num_train_timesteps"],
                beta_start=sched_params["beta_start"],  # 0.0001
                beta_end=sched_params["beta_end"],  # 0.02
                beta_schedule=sched_params["beta_schedule"],
                trained_betas=None,
                clip_sample=False,
                clip_sample_range=1.0,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type=sched_params["prediction_type"],
                thresholding=False,
                dynamic_thresholding_ratio=0.995,
                sample_max_value=1.0)
        else:
            raise NotImplementedError(sched_params["name"])
        if sched_params["name"] in ("ddpm", "ddim"):
            self.pred_type = sched_params["prediction_type"]
        else:
            raise NotImplementedError(sched_params["name"])
        self.sched_params = sched_params

        # Self-conditioning.
        self.use_sc = False if sc_params is None else sc_params["use"]
        if self.use_sc:
            self.sc_train_p = sc_params["train_p"]
        else:
            self.sc_train_p = None

        # xyz loss (by default it is not used).
        self.xyz_loss = None


    def sample_time(self, x0, batch_size):
        if self.sched_params["name"] in ("ddpm", "ddim"):
            t = torch.randint(0, self.sched_params["num_train_timesteps"],
                              (batch_size,),
                              device=x0.device, dtype=torch.long)
        else:
            raise KeyError(self.sched_params["name"])
        return t


    def loss(self,
             batch,
             noise: Optional[torch.Tensor] = None,
             reduction: str = "mean"):
        """
        TODO.
        """

        # Get batch size
        batch_size = batch.num_graphs
        # Encodings at time zero.
        x0 = batch.z

        # Get random $t$ for each sample in the batch.
        t = self.sample_time(x0, batch_size)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is not None:
            raise NotImplementedError()
        noise = torch.randn_like(batch.z)

        # Sample $x_t$ for $q(x_t|x_0)$.
        xt = self.sched.add_noise(x0, noise, t)

        # Regular diffusion modeling.
        if not self.use_sc:
            # Get the model output.
            model_out = self.sched.scale_model_input(
                sample=self.eps_model(xt=xt, t=t, batch=batch),
                timestep=t)
            

        # Diffusion modeling with self-conditioning.
        else:
            raise NotImplementedError()

        if self.pred_type == "epsilon":
            target = noise
        else:
            raise NotImplementedError()

        # Compute the main, MSE-based loss.
        if self.loss_type == 'l1':
            loss = F.l1_loss(target, model_out, reduction=reduction)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, model_out, reduction=reduction)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(target, model_out, reduction=reduction)
        else:
            raise NotImplementedError()

        # Compute the xyz structure loss.
        if self.xyz_loss is not None and t.min() < self.xyz_loss_t_on:
            """
            xyz_loss = self.compute_xyz_loss(xt, x0, t, eps_theta, batch)
            loss += xyz_loss
            """
            raise NotImplementedError()

        return loss


    @torch.no_grad()
    def sample(self, batch, x_0=None, t_start=None,
               n_steps=None, variance=None,
               *args, **kwargs):
        
        if x_0 is not None or t_start is not None:
            raise NotImplementedError()
        
        if self.sched_params["name"] == "ddpm":
            if n_steps is not None:
                self.sched.set_timesteps(n_steps)
            if variance is not None:
                # Options: ("fixed_small", "fixed_small_log",
                #           "fixed_large", "fixed_large_log")
                self.sched.variance_type = variance
        elif self.sched_params["name"] == "ddim":
            self.sched.set_timesteps(n_steps if n_steps is not None \
                                     else self.sched.config.num_train_timesteps)
        else:
            raise NotImplementedError()

        if self.pred_type != "epsilon":
            raise NotImplementedError()

        model = self.get_sample_model()
        x_t = torch.randn_like(batch.z)
        batch_size = batch.num_graphs
        for i in self.sched.timesteps:
            t = torch.full((batch_size, ), i,
                           device=batch.z.device, dtype=torch.long)
            with torch.no_grad():
                noisy_residual = self.sched.scale_model_input(
                    sample=model(xt=x_t, t=t, batch=batch),
                    timestep=t)
            x_t = self.sched.step(noisy_residual, i, x_t).prev_sample
        out = x_t
        return out
