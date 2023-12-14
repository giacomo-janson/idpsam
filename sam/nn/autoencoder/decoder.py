import math
from typing import Tuple, Union, Callable
import inspect

import numpy as np

import torch
import torch.nn as nn

from sam.nn.common import get_act_fn, AF2_PositionalEmbedding
from sam.nn.autoencoder.ca_models import AE_IdpGAN_TransformerBlock


################################################################################
# Decoder network.                                                             #
################################################################################

class CA_TransformerDecoder(nn.Module):

    def __init__(
        self,
        encoding_dim: int = 16,
        output_dim: int = 3,
        use_input_mlp: bool = False,
        num_layers: int = 16,
        attention_type: str = "transformer",
        embed_dim: int = 256,
        d_model: int = None,
        num_heads: int = 20,
        mlp_dim: int = None,
        dropout: int = None,
        norm_eps: float = 1e-5,
        norm_pos: str = "pre",
        activation: Union[str, Callable] = "relu",
        bead_embed_dim: int = 32,
        pos_embed_dim: int = 64,
        use_bias_2d: bool = True,
        pos_embed_r: int = 32,
        use_res_ids: bool = False,
        embed_inject_mode: str = "adanorm"
        # cg: int = None
        # input_inject_mode: str = None,
        # input_inject_pos: str = "out",
        ):
        """
        `encoding_dim`: dimension of the structural encoding vectors.
        """

        super().__init__()

        ### Check and store the attributes.
        self.embed_inject_mode = embed_inject_mode
        self.use_res_ids = use_res_ids
        self.embed_dim = embed_dim
        # self.cg = cg

        ### Shared functions.
        if isinstance(activation, str):
            act_cls = get_act_fn(activation_name=activation)
        else:
            raise TypeError(activation.__class__)
        
        ### Process input.
        if not use_input_mlp:
            self.project_input = nn.Linear(encoding_dim, embed_dim)
        else:
            self.project_input = nn.Sequential(nn.Linear(encoding_dim,
                                                         embed_dim),
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
            dim_order="transformer")

        ### Transformer layers.
        if mlp_dim is None:
            mlp_dim = 4 * embed_dim
        
        self.layers = []
        for l in range(num_layers):
            
            layer_l = AE_IdpGAN_TransformerBlock(
                embed_dim=embed_dim,
                embed_2d_dim=None,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                norm_pos=norm_pos,
                activation=act_cls,
                d_model=d_model,
                add_bias_kv=False,
                embed_inject_mode=embed_inject_mode,
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
        self.out_module = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                        act_cls(),
                                        nn.Linear(embed_dim, output_dim))
        # if self.cg is None:
        #     pass
        # else:
        #     self.out_module = nn.Sequential(nn.Linear(embed_dim, embed_dim),
        #                                     act_cls(),
        #                                     nn.Linear(embed_dim, 3*self.cg))
        # self.out_conv = nn.ConvTranspose1d(
        #     in_channels=embed_dim, out_channels=embed_dim,
        #     kernel_size=self.cg,
        #     stride=2, padding=0, output_padding=0, groups=1, bias=True,
        #     dilation=1, padding_mode='zeros')


    def get_embed_dim(self):
        return self.embed_dim


    def forward(self, x, a=None, r=None):
        """
        x: input tensor with shape (B, L, E).
        a: amino acid tensor with shape (B, L). It should be set to 'None' if
            'embed_inject_mode' is also 'None'.
        """
        ### Input.
        x = self.project_input(x).transpose(0, 1)

        ### Bead and time embeddings.
        if self.embed_inject_mode is not None:
            a_e = self.beads_embed(a).transpose(0, 1)
        else:
            a_e = None

        ### Positional embeddings.
        p = self.embed_pos(x, r=r)
        
        # ### Go through all the transformer blocks.
        x_0 = None
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x=x, a=a_e, p=p, z=None, x_0=x_0)

        ### Output module.
        x = self.out_module(x)
        xyz = x.transpose(0, 1)  # (L, B, 3) => (B, L, 3)
        return xyz


    def nn_forward(self, e, batch, num_nodes=None):
        if num_nodes is not None:
            raise NotImplementedError()
        a = batch.a if self.embed_inject_mode is not None else None
        r = batch.r if self.use_res_ids else None
        xyz = self.forward(x=e, a=a, r=r)
        return xyz


def get_decoder(model_cfg, input_dim=None, output_dim=None):
    # Get the arguments of the eps network class.
    args = list(
        inspect.signature(CA_TransformerDecoder.__init__).parameters.keys())
    args.remove("encoding_dim")
    # Get from 'model_cfg' the corresponding arguments.
    params = {}
    for arg in args:
        if arg in model_cfg["decoder"]:
            params[arg] = model_cfg["decoder"][arg]
    # Initialize the network.
    return CA_TransformerDecoder(
        encoding_dim=input_dim if input_dim is not None \
                     else model_cfg["generative_model"]["encoding_dim"],
        output_dim=output_dim if output_dim is not None else 3,
        **params)


if __name__ == "__main__":
    
    torch.manual_seed(0)

    # Batch size.
    N = 128
    # Number of residues (sequence length).
    L = 19
    # Encoding dimension.
    e_dim = 16

    s = torch.randn(N, L, e_dim)
    a = torch.randint(0, 20, (N, L))

    net = CA_TransformerDecoder(
        encoding_dim=e_dim,
        use_input_mlp=True,
        num_layers=4,
        attention_type="timewarp",
        embed_dim=256,
        d_model=512,
        num_heads=32,
        mlp_dim=None,
        dropout=None,
        norm_eps=1e-5,
        norm_pos="post",
        activation="gelu",
        bead_embed_dim=None,
        pos_embed_dim=64,
        use_bias_2d=True,
        pos_embed_r=32,
        embed_inject_mode=None,
        # input_inject_mode="adanorm",
        # input_inject_pos="out",
    )
    out = net(x=s, a=a)

    print(out.shape)
