from sam.diffusion.diffusers_dm import Diffusers


def get_diffusion_model(model_cfg, network, ema=None):
    model_type = model_cfg["latent_generative_model"]["type"]
    if model_type == "diffusers_dm":
        diffusion = Diffusers(
            eps_model=network,
            sched_params=model_cfg["latent_generative_model"]["sched_params"],
            loss=model_cfg["latent_generative_model"].get("loss", "l2"),
            ema=ema,
            sc_params=None
        )
    else:
        raise KeyError(model_type)
    return diffusion