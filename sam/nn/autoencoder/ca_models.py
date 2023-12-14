from typing import Callable
import math
import numpy as np
import torch
import torch.nn as nn

from sam.nn.transformer import TransformerLayer as TransformerLayerIdpGAN
from sam.nn.transformer import TransformerTimewarpLayer


################################################################################
# Transformer block.                                                           #
################################################################################

class AE_IdpGAN_TransformerBlock(nn.Module):
    """Transformer layer block from idpGAN."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        embed_2d_dim: int = None,
        norm_pos: str = "pre",
        activation: Callable = nn.ReLU,
        d_model: int = None,
        add_bias_kv: bool = True,
        embed_inject_mode: str = "adanorm",
        embed_2d_inject_mode: str = None,
        bead_embed_dim: int = 32,
        pos_embed_dim: int = 64,
        use_bias_2d: int = True,
        attention_type: str = "transformer"
        # input_inject_mode: str = None,
        # input_inject_pos: str = "out",
    ):

        ### Initialize and store the attributes.
        super().__init__()

        if d_model is None:
            d_model = embed_dim
        if not norm_pos in ("pre", "post"):
            raise KeyError(norm_pos)
        self.norm_pos = norm_pos
        self.attention_type = attention_type

        ### Transformer layer.
        # Edge features (2d embeddings).
        if embed_2d_dim is not None:
            if embed_2d_inject_mode == "add":
                attn_in_dim_2d = embed_2d_dim
                if embed_2d_dim != pos_embed_dim:
                    self.project_pos_embed_dim = nn.Linear(pos_embed_dim,
                                                           embed_2d_dim)
                else:
                    self.project_pos_embed_dim = nn.Identity()
            elif embed_2d_inject_mode == "concat":
                attn_in_dim_2d = embed_2d_dim + pos_embed_dim
            elif embed_2d_inject_mode is None:
                raise ValueError(
                    "Please provide a `embed_2d_inject_mode` when using"
                    " `embed_2d_dim` != 'None'")
            else:
                raise KeyError(embed_2d_inject_mode)
            self.embed_2d_inject_mode = embed_2d_inject_mode
        else:
            self.embed_2d_inject_mode = None
            attn_in_dim_2d = pos_embed_dim

        # Actual transformer layer.
        self.attn_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=embed_inject_mode != "adanorm")
        if attention_type == "transformer":
            self.self_attn = TransformerLayerIdpGAN(
                in_dim=embed_dim,
                d_model=d_model,
                nhead=num_heads,
                dp_attn_norm="d_model",  # dp_attn_norm="head_dim",
                in_dim_2d=attn_in_dim_2d,
                use_bias_2d=use_bias_2d)
        elif attention_type  == "timewarp":
            self.self_attn = TransformerTimewarpLayer(
                in_dim=embed_dim,
                d_model=d_model,
                nhead=num_heads,
                in_dim_2d=attn_in_dim_2d,
                use_bias_2d=use_bias_2d)
        else:
            raise KeyError(attention_type)
        
        ### MLP.
        if embed_inject_mode is not None:
            if embed_inject_mode == "concat" and self.norm_pos == "post":
                # IdpGAN mode.
                fc1_in_dim = embed_dim + bead_embed_dim
            else:
                fc1_in_dim = embed_dim
        else:
            fc1_in_dim = embed_dim
        self.fc1 = nn.Linear(fc1_in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.final_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=embed_inject_mode != "adanorm")
        self.act = activation()

        ### Conditional information injection module.
        self.cond_injection_module = AE_ConditionalInjectionModule(
            mode=embed_inject_mode,
            embed_dim=embed_dim,
            bead_embed_dim=bead_embed_dim,
            activation=activation,
            norm_pos=norm_pos
        )


    def forward(self, x, a, p, z=None, x_0=None):

        # Attention mechanism.
        residual = x
        inj_out = self.cond_injection_module(a=a)
        x = self.cond_injection_module.inject_0(x, inj_out)
        if self.norm_pos == "pre":
            x = self.attn_norm(x)
        x = self.cond_injection_module.inject_1_pre(x, inj_out)
        if self.embed_2d_inject_mode == "add":
            z_hat = z + self.project_pos_embed_dim(p)
        elif self.embed_2d_inject_mode == "concat":
            z_hat = torch.cat([z, p], axis=3)
        elif self.embed_2d_inject_mode is None:
            z_hat = p
        else:
            raise KeyError(self.embed_2d_inject_mode)
        x = self.self_attn(x, x, x, p=z_hat)[0]
        attn = None
        x = self.cond_injection_module.inject_1_post(x, inj_out)
        x = residual + x
        if self.norm_pos == "post":
            x = self.attn_norm(x)

        # MLP update.
        residual = x
        if self.norm_pos == "pre":
            x = self.final_norm(x)
        x = self.cond_injection_module.inject_2_pre(x, inj_out)
        x = self.fc2(self.act(self.fc1(x)))
        x = self.cond_injection_module.inject_2_post(x, inj_out)
        x = residual + x
        if self.norm_pos == "post":
            x = self.final_norm(x)

        # # Inject initial input.
        # x = self.inject_input(x, x_0, pos="out")

        return x, attn


###########################################################################
# Conditional information injection (amino acid embeddings).              #
###########################################################################

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class AE_ConditionalInjectionModule(nn.Module):

    def __init__(self,
                 mode: str,
                 embed_dim: int = 256,
                 bead_embed_dim: int = 256,
                 activation: Callable = nn.SiLU,
                 # mlp_ratio: float = 4.0,
                 norm_pos: str = "pre"):
        """
        `mlp_ratio`: used for with `adanorm`.
        """

        super().__init__()
        self.mode = mode
        if self.mode == "adanorm":
            self.adaLN_modulation = nn.Sequential(
                activation(),
                nn.Linear(bead_embed_dim, 6 * embed_dim, bias=True)
            )
        elif self.mode == "concat": 
            self.norm_pos = norm_pos
            if self.norm_pos == "pre":
                self.project_concat = nn.Linear(embed_dim+bead_embed_dim,
                                                embed_dim)
        elif self.mode is None:
            pass
        else:
            raise KeyError(mode)
    
    def initialize_weights(self):
        if self.mode == "adanorm":
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        elif self.mode == "concat": 
            pass
        elif self.mode is None:
            pass
        else:
            raise KeyError(self.mode)
    
    def forward(self, a):
        if self.mode == "adanorm":
            c = a
            adaLN_m_r = self.adaLN_modulation(c).chunk(6, dim=2)
            out = {"shift_msa": adaLN_m_r[0],
                   "scale_msa": adaLN_m_r[1],
                   "gate_msa": adaLN_m_r[2],
                   "shift_mlp": adaLN_m_r[3],
                   "scale_mlp": adaLN_m_r[4],
                   "gate_mlp": adaLN_m_r[5]}
        elif self.mode == "concat": 
            out = {"a": a}
        elif self.mode is None:
            out = {"a": a}
        else:
            raise KeyError(self.mode)
        return out
    
    def inject_0(self, x, inj_out):
        if self.mode == "adanorm":
            return x
        elif self.mode == "concat":
            if self.norm_pos == "post":
                return x
            return self.project_concat(torch.cat([x, inj_out["a"]], axis=2))
        elif self.mode is None:
            return x
        else:
            raise KeyError(self.mode)

    def inject_1_pre(self, x, inj_out):
        if self.mode == "adanorm":
            return modulate(x, inj_out["shift_msa"], inj_out["scale_msa"])
        elif self.mode == "concat":
            return x
        elif self.mode is None:
            return x
        else:
            raise KeyError(self.mode)

    def inject_1_post(self, x, inj_out):
        if self.mode == "adanorm":
            return x * inj_out["gate_msa"]
        elif self.mode == "concat":
            return x
        elif self.mode is None:
            return x
        else:
            raise KeyError(self.mode)

    def inject_2_pre(self, x, inj_out):
        if self.mode == "adanorm":
            return modulate(x, inj_out["shift_mlp"], inj_out["scale_mlp"])
        elif self.mode == "concat":
            if self.norm_pos == "pre":
                return x
            return torch.cat([x, inj_out["a"]], axis=2)
        elif self.mode is None:
            return x
        else:
            raise KeyError(self.mode)

    def inject_2_post(self, x, inj_out):
        if self.mode == "adanorm":
            return x * inj_out["gate_mlp"]
        elif self.mode == "concat":
            return x
        elif self.mode is None:
            return x
        else:
            raise KeyError(self.mode)
