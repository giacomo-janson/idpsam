import math
from typing import Tuple, Union, Callable
import inspect

import numpy as np

import torch
import torch.nn as nn

from sam.coords import calc_dmap, torch_chain_dihedrals
from sam.nn.common import get_act_fn, AF2_PositionalEmbedding
from sam.nn.geometric import GaussianSmearing, ExpNormalSmearing
from sam.nn.autoencoder.ca_models import AE_IdpGAN_TransformerBlock


################################################################################
# Functions.                                                                   #
################################################################################

def get_chain_torsion_features(x):
    t = torch_chain_dihedrals(x).unsqueeze(2)
    t_f = torch.cat([torch.cos(t), torch.sin(t), torch.ones_like(t)],
                    dim=2)
    t_f = nn.functional.pad(t_f, (0, 0, 1, 2))
    return t_f


################################################################################
# Encoder network.                                                             #
################################################################################

class CA_TransformerEncoder(nn.Module):

    def __init__(
        self,
        encoding_dim: int = 16,
        num_layers: int = 16,
        attention_type: str = "transformer",
        embed_dim: int = 256,
        embed_2d_dim: int = 128,
        d_model: int = None,
        num_heads: int = 20,
        mlp_dim: int = None,
        dropout: int = None,
        norm_eps: float = 1e-5,
        norm_pos: str = "pre",
        activation: Union[str, Callable] = "relu",
        out_mode: str = "idpgan",
        out_conv: Union[list, tuple, str] = None,
        bead_embed_dim: int = 32,
        pos_embed_dim: int = 64,
        use_bias_2d: bool = True,
        pos_embed_r: int = 32,
        use_res_ids: bool = False,
        embed_inject_mode: str = "adanorm",
        embed_2d_inject_mode: str = "concat",
        dmap_ca_min: float = 0.0,
        dmap_ca_cutoff: float = 10.0,
        dmap_ca_num_gaussians: int = 128,
        dmap_embed_type: str = "rbf",
        dmap_embed_trainable: bool = False,
        use_dmap_mlp: bool = False,
        # cg: int = None,
        # cg_upscale: int = 2,
        ):
        """
        `encoding_dim`: dimension of the structural encoding vectors.
        `dmap_ca_min`: min distance in the radial basis function (RBF) embedding
            for Ca-Ca distances. Values are in Amstrong.
        `dmap_ca_cutoff`: maximum distance in the RBF ebbedding for Ca-Ca
            distances.
        """

        super().__init__()

        ### Check and store the attributes.
        self.embed_dim = embed_dim
        self.embed_inject_mode = embed_inject_mode
        self.use_res_ids = use_res_ids
        # self.cg = cg

        ### Shared functions.
        if isinstance(activation, str):
            act_cls = get_act_fn(activation_name=activation)
        else:
            raise TypeError(activation.__class__)
        self.act_cls = act_cls

        ### Process input.
        # Embed Ca-Ca distances.
        if dmap_embed_type == "rbf":
            self.dmap_ca_expansion = GaussianSmearing(
                start=dmap_ca_min,
                stop=dmap_ca_cutoff,
                num_gaussians=dmap_ca_num_gaussians)
        elif dmap_embed_type == "expnorm":
            self.dmap_ca_expansion = ExpNormalSmearing(
                cutoff_lower=dmap_ca_min,
                cutoff_upper=dmap_ca_cutoff,
                num_rbf=dmap_ca_num_gaussians,
                trainable=dmap_embed_trainable)
        else:
            raise KeyError(dmap_embed_type)
        if not use_dmap_mlp:
            self.project_dmap = nn.Sequential(nn.Linear(dmap_ca_num_gaussians,
                                                        embed_2d_dim))
        else:
            self.project_dmap = nn.Sequential(nn.Linear(dmap_ca_num_gaussians,
                                                        embed_2d_dim),
                                              act_cls(),
                                              nn.Linear(embed_2d_dim,
                                                        embed_2d_dim))
        # Embed torsions.
        # Three dimensions for: cos(a_i), sin(a_i), mask(i)
        self.project_input = nn.Sequential(nn.Linear(3, embed_dim),
                                            act_cls(),
                                            nn.Linear(embed_dim, embed_dim))

        ### Amino acid embedding.
        if embed_inject_mode is not None:
            self.beads_embed = nn.Embedding(20, bead_embed_dim,
                                            # padding_idx=self.padding_idx,
                                           )

        ### Positional embeddings.
        self.embed_pos = AF2_PositionalEmbedding(
            pos_embed_dim=pos_embed_dim,
            pos_embed_r=pos_embed_r,
            dim_order="trajectory")

        ### Transformer layers.
        if mlp_dim is None:
            mlp_dim = 4 * embed_dim
        
        self.layers = []
        for l in range(num_layers):
            
            layer_l = AE_IdpGAN_TransformerBlock(
                embed_dim=embed_dim,
                embed_2d_dim=embed_2d_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                norm_pos=norm_pos,
                activation=act_cls,
                d_model=d_model,
                add_bias_kv=False,
                embed_inject_mode=embed_inject_mode,
                embed_2d_inject_mode=embed_2d_inject_mode,
                bead_embed_dim=bead_embed_dim,
                pos_embed_dim=pos_embed_dim,
                use_bias_2d=use_bias_2d,
                attention_type=attention_type,
                # input_inject_mode=input_inject_mode,
                # input_inject_pos=input_inject_pos,
            )

            self.layers.append(layer_l)
            
        self.layers = nn.ModuleList(self.layers)

        ### Output module.
        if out_mode in ("idpgan", "idpgan_norm"):
            self.out_module = [nn.Linear(embed_dim, embed_dim),
                               act_cls(),
                               nn.Linear(embed_dim, encoding_dim)]
            if out_mode == "idpgan_norm":
                self.out_module.append(nn.LayerNorm(encoding_dim,
                                                    elementwise_affine=False))
            
            self.out_module = nn.Sequential(*self.out_module)
            if out_conv is not None:
                # self.out_conv_layer = SmoothConv(out_conv, encoding_dim)
                raise NotImplementedError()
            else:
                self.out_conv_layer = None
        # elif out_mode == "esm":
        #     self.out_module = nn.Sequential(# nn.LayerNorm(),
        #                                     nn.Linear(embed_dim, embed_dim),
        #                                     act_cls(),
        #                                     nn.LayerNorm(embed_dim),
        #                                     nn.Linear(embed_dim, input_dim))
        else:
            raise KeyError(out_mode)


    def get_embed_dim(self):
        return self.embed_dim


    def forward(self, x, a=None, r=None):
        """
        x: input tensor with shape (B, L, 3).
        a: amino acid tensor with shape (B, L).
        """
        ### Input.
        # Get the Ca-Ca distance matrix (2d features).
        dmap_ca = calc_dmap(x)
        rbf_ca = self.dmap_ca_expansion(dmap_ca).transpose(1, 3)
        z = self.project_dmap(rbf_ca)

        # Amino acid encodings.
        if self.embed_inject_mode is not None:
            a_e = self.beads_embed(a).transpose(0, 1)
        else:
            a_e = None
        
        # Torsion angles as 1d input embeddings.
        s = self.project_input(get_chain_torsion_features(x))
        s = s.transpose(0, 1)  # (B, L, E)

        ### Positional embeddings.
        p = self.embed_pos(x, r=r)

        ### Go through all the transformer blocks.
        s_0 = None
        for layer_idx, layer in enumerate(self.layers):
            s, attn = layer(x=s, a=a_e, p=p, z=z, x_0=s_0)

        ### Output module.
        s = self.out_module(s)
        enc = s.transpose(0, 1)  # (L, B, E) => (B, L, E)
        # if self.out_conv_layer is not None:
        #     enc = self.out_conv_layer(enc.transpose(1, 2)).transpose(2, 1)

        return enc

    
    def nn_forward(self, batch, x=None, num_nodes=None):
        if num_nodes is not None:
            raise NotImplementedError()
        a = batch.a if self.embed_inject_mode is not None else None
        r = batch.r if self.use_res_ids else None
        enc = self.forward(x=batch.x if x is None else x,
                           a=a, r=r)
        return enc

    
# def get_network(output_dim, model_params, stage):
#     # Get the arguments of the eps network class.
#     args = list(inspect.signature(CA_TransformerEncoder.__init__).parameters.keys())
#     args.remove("encoding_dim")
#     # Get from 'model_params' the corresponding arguments.
#     params = {}
#     for arg in args:
#         if arg in model_params["stage_0"]["enc"]:
#             params[arg] = model_params["stage_0"]["enc"][arg]
#     # Initialize the network.
#     return CA_TransformerEncoder(
#         encoding_dim=output_dim,  # model_params["generative_model"]["encoding_dim"]
#         **params)


if __name__ == "__main__":

    torch.manual_seed(0)

    # Batch size.
    N = 128
    # Number of residues (sequence length).
    L = 19
    # Encoding dimension.
    e_dim = 16

    x = torch.randn(N, L, 3)
    a = torch.randint(0, 20, (N, L))

    net = CA_TransformerEncoder(
        encoding_dim=e_dim,
        num_layers=5,
        # attention_type="transformer",
        embed_dim=256,
        embed_2d_dim=192,
        d_model=None,
        num_heads=8,
        mlp_dim=256,
        dropout=None,
        norm_eps=1e-5,
        norm_pos="pre",
        activation="gelu",
        out_mode="idpgan_norm",
        # out_conv=[0.25, 0.5, 0.25],
        bead_embed_dim=32,
        pos_embed_dim=64,
        use_bias_2d=True,
        pos_embed_r=32,
        use_res_ids=True,
        embed_inject_mode="concat",
        embed_2d_inject_mode="concat",
        dmap_ca_min=0.0,
        dmap_ca_cutoff=3.0,
        dmap_ca_num_gaussians=320,
        dmap_embed_type="rbf",
        use_dmap_mlp=True)

    out = net(x=x, a=a)

    print(out.shape)
