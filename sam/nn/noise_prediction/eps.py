import math
from typing import Tuple, Union, Callable
import inspect
import numpy as np
import torch
import torch.nn as nn
from sam.nn.common import get_act_fn, AF2_PositionalEmbedding, MLP
from sam.nn.noise_prediction.embedding import (TimestepEmbedder,
                                               ConditionalInjectionModule,
                                               InputInjectionModule)
from sam.nn.transformer import TransformerLayer as TransformerLayerIdpGAN
from sam.nn.transformer import PyTorchAttentionLayer
# from sam_lib.nn_models.esm_layers import ESM_AttentionLayer
# from sam_lib.pdm.nn_models.common import prepare_eps_input


################################################################################
# Transformer block.                                                           #
################################################################################

class IdpGAN_TransformerBlock(nn.Module):
    """Transformer layer block from idpGAN."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        norm_pos: str = "pre",
        activation: Callable = nn.ReLU,
        d_model: int = None,
        add_bias_kv: bool = False,
        attention_type: str = "transformer",
        embed_inject_mode: str = "adanorm",
        bead_embed_dim: int = 256,
        time_embed_dim: int = 256,
        pos_embed_dim: int = 128,
        edge_update_mode: str = None,
        edge_update_params: dict = {},
        use_bias_2d: int = True,
        input_dim: int = 16,
        input_inject_mode: str = None,
        input_inject_pos: str = "out",
        # tem_inject_mode: str = None,
        # tem_inject_pos: str = "out"
    ):
        super().__init__()

        if d_model is None:
            d_model = embed_dim
        if not norm_pos in ("pre", "post"):
            raise KeyError(norm_pos)
        self.norm_pos = norm_pos

        ### Transformer layer.
        self.attn_norm = nn.LayerNorm(
            embed_dim, elementwise_affine=embed_inject_mode != "adanorm")

        if attention_type == "transformer":
            self.self_attn = TransformerLayerIdpGAN(
                in_dim=embed_dim,
                d_model=d_model,
                nhead=num_heads,
                dp_attn_norm="d_model",
                in_dim_2d=pos_embed_dim,
                use_bias_2d=use_bias_2d)
        elif attention_type == "pytorch":
            self.self_attn = PyTorchAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                edge_dim=pos_embed_dim,
                add_bias_kv=add_bias_kv,
                use_bias_2d=use_bias_2d,
                dropout=0.0)
        else:
            raise KeyError(attention_type)
        
        ### MLP.
        if embed_inject_mode == "concat" and self.norm_pos == "post":
            # IdpGAN mode.
            if bead_embed_dim is not None:  # Amino acid conditional model.
                fc1_in_dim = embed_dim + bead_embed_dim + time_embed_dim
            else:  # Amino acid unconditional model.
                fc1_in_dim = embed_dim + time_embed_dim
        else:
            fc1_in_dim = embed_dim
        self.fc1 = nn.Linear(fc1_in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.final_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=embed_inject_mode != "adanorm")
        self.act = activation()

        ### Conditional information injection module.
        self.cond_injection_module = ConditionalInjectionModule(
            mode=embed_inject_mode,
            embed_dim=embed_dim,
            bead_embed_dim=bead_embed_dim,
            time_embed_dim=time_embed_dim,
            activation=activation,
            norm_pos=norm_pos
        )

        ### Edge representation update mode.
        edge_updater = get_edge_updater(
            embed_dim=embed_dim,
            edge_dim=pos_embed_dim,
            edge_update_mode=edge_update_mode,
            edge_update_params=edge_update_params,
            activation=activation
        )
        if edge_update_mode is not None:
            self.edge_updater = edge_updater

        self.edge_update_mode = edge_update_mode

        ### Input injection module.
        if input_inject_mode not in (None, "add", "adanorm"):
            raise KeyError(input_inject_mode)
        self.input_inject_mode = input_inject_mode

        if input_inject_pos not in ("out", ):
            raise KeyError(input_inject_pos)
        self.input_inject_pos = input_inject_pos

        if input_inject_mode is None:
            pass
        elif input_inject_mode == "add":
            # TODO: it should be included in the 'InputInjectionModule' module.
            self.input_project = nn.Linear(input_dim, embed_dim)
            self.input_inject_norm = nn.LayerNorm(embed_dim)
        elif input_inject_mode == "adanorm":
            self.input_inject_module = InputInjectionModule(
                mode=input_inject_mode,
                input_dim=input_dim,
                embed_dim=embed_dim,
                time_embed_dim=time_embed_dim,
                activation=activation)
        else:
            raise KeyError(input_inject_mode)
        
        # ### Template injection module.
        # if tem_inject_mode not in (None, "add", "adanorm"):
        #     raise KeyError(tem_inject_mode)
        # self.tem_inject_mode = tem_inject_mode

        # if tem_inject_pos not in ("out", ):
        #     raise KeyError(tem_inject_pos)
        # self.tem_inject_pos = tem_inject_pos

        # if tem_inject_mode is None:
        #     pass
        # elif tem_inject_mode in ("add", "adanorm"):
        #     self.tem_inject_module = InputInjectionModule(
        #         mode=tem_inject_mode,
        #         input_dim=input_dim,
        #         embed_dim=embed_dim,
        #         time_embed_dim=time_embed_dim,
        #         activation=activation)
        # else:
        #     raise KeyError(tem_inject_mode)


    def forward(self, x, t, p, a=None, x_0=None, x_tem=None):

        # Attention mechanism.
        residual = x
        inj_out = self.cond_injection_module(a=a, t=t)
        x = self.cond_injection_module.inject_1_proto(x, inj_out)
        if self.norm_pos == "pre":
            x = self.attn_norm(x)
        x = self.cond_injection_module.inject_1_pre(x, inj_out)
        x = self.self_attn(x, x, x, p=p)[0]
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

        # Inject initial input.
        x = self.inject_input(x, x_0, t, pos="out")

        # # Inject template.
        # x = self.inject_template(x, x_tem, t, pos="out")

        # Edge update.
        p = self.update_edges(x, p)

        return x, p


    def inject_input(self, x, x_0, t, pos):
        self._check_pos(pos)
        if pos != self.input_inject_pos:
            return x
        # if x_0 is None:
        #     raise TypeError()
        # Do not perform any operation.
        if self.input_inject_mode is None:
            pass
        # Add projection of input to embeddings.
        elif self.input_inject_mode == "add":
            x = x + self.input_project(x_0)
            x = self.input_inject_norm(x)
        elif self.input_inject_mode == "adanorm":
            x = self.input_inject_module(x, x_0, t)
        else:
            raise KeyError(self.input_inject_mode)
        return x
    

    # def inject_template(self, x, x_tem, t, pos):
    #     self._check_pos(pos)
    #     if pos != self.tem_inject_pos:
    #         return x
    #     # if x_tem is None:
    #     #     raise TypeError()
    #     # Do not perform any operation.
    #     if self.tem_inject_mode is None:
    #         pass
    #     # Actually inject template information.
    #     elif self.tem_inject_mode in ("add", "adanorm"):
    #         x = self.tem_inject_module(x, x_tem, t)
    #     else:
    #         raise KeyError(self.tem_inject_mode)
    #     return x


    def update_edges(self, x, z):
        if self.edge_update_mode is None:
            return z
        elif self.edge_update_mode in ("framediff", "framediff_add", "sam_0", ):
            return self.edge_updater(x, z)
        else:
            raise KeyError(self.edge_update_mode)

    def _check_pos(self, pos):
        if pos not in ("out", ):
            raise KeyError(pos)


################################################################################
# Layers.                                                                      #
################################################################################

class EPS_EdgeEmbedder(nn.Module):
    """Embed input edge features."""

    def __init__(self,
                 mode: str,
                 edge_dim: int,
                 pos_embed_dim: int,
                 time_embed_dim: int,
                 bead_embed_dim: int,
                 activation: callable = nn.ReLU):
        super().__init__()
        if not mode in ("concat", ):
            raise KeyError(mode)
        self.mode = mode
        self.bead_embed_dim = bead_embed_dim
        if mode == "concat":
            in_dim = pos_embed_dim + time_embed_dim
            if bead_embed_dim is not None:
                in_dim += bead_embed_dim*2
        else:
            raise KeyError(mode)
        self.mlp = MLP(in_dim=in_dim, out_dim=edge_dim, activation=activation,
                       n_hidden_layers=0, final_norm=False)

    def forward(self, p, t, a):
        """
        p: 2d positional embedding of shape (B, L, L, E_p).
        t: 1d time embedding of shape (L, B, E_t).
        a: 1d bead embedding of shape (L, B, E_b). Only used if `bead_embed_dim`
           is set to 'None'.
        """
        if self.mode == "concat":
            x_in = [p,
                    t.transpose(0, 1).unsqueeze(2).repeat(1, 1, p.shape[1], 1)]
            if self.bead_embed_dim is not None:
                a_in = a.transpose(0, 1).unsqueeze(2).repeat(1, 1, p.shape[1], 1)
                x_in.extend([
                    a_in,
                    a_in.transpose(1, 2)
                ])
            x_in = torch.cat(x_in, dim=3)
        else:
            raise KeyError(mode)
        return self.mlp(x_in)


class EdgeUpdaterFrameDiff(nn.Module):
    """Edge representation updater from FrameDiff."""

    def __init__(self, node_dim, edge_dim, outer_operation="concat",
                 activation=nn.ReLU):
        super().__init__()
        self.down_linear = nn.Linear(node_dim, node_dim // 2)
        if outer_operation == "concat":
            edge_mlp_dim = node_dim + edge_dim  # (embed_dim // 2) * 2 + edge_dim
        elif outer_operation == "add":
            edge_mlp_dim = node_dim // 2 + edge_dim
            # self.post_add_norm = nn.LayerNorm(node_dim // 2)
        else:
            raise KeyError(outer_operation)
        self.outer_operation = outer_operation
        self.edge_mlp = nn.Sequential(nn.Linear(edge_mlp_dim, edge_mlp_dim),
                                      activation(),
                                      nn.Linear(edge_mlp_dim, edge_mlp_dim),
                                      activation())

        self.edge_out_linear = nn.Linear(edge_mlp_dim, edge_dim)
        self.edge_out_norm = nn.LayerNorm(edge_dim)

    def forward(self, x, z):
        x_down = self.down_linear(x).transpose(0, 1)
        num_res = x_down.shape[1]
        if self.outer_operation == "concat":
            # x_down = x_down.unsqueeze(2).repeat(1, 1, x_down.shape[1], 1)
            # z_in = torch.cat([x_down, x_down.transpose(1, 2), z], dim=3)
            edge_bias = torch.cat([
                torch.tile(x_down[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(x_down[:, None, :, :], (1, num_res, 1, 1)),
            ], axis=-1)
        elif self.outer_operation == "add":
            edge_bias = torch.tile(x_down[:, :, None, :], (1, 1, num_res, 1)) + \
                        torch.tile(x_down[:, None, :, :], (1, num_res, 1, 1))
        else:
            raise KeyError(self.outer_operation)
        z_in = torch.cat([edge_bias, z], axis=-1)
        z = self.edge_out_linear(self.edge_mlp(z_in) + z_in)
        z = self.edge_out_norm(z)
        return z


class EdgeUpdaterSAM_0(nn.Module):
    """Custom edge representation updater 0."""

    def __init__(self,
                 node_dim, edge_dim,
                 edge_downsample=2,
                 activation=nn.ReLU):
        super().__init__()
        hidden_dim = edge_dim // edge_downsample
        self.node_input_linear = nn.Linear(node_dim, hidden_dim)
        self.edge_input_linear = nn.Linear(edge_dim, hidden_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 activation(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 activation())
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        self.edge_out_linear = nn.Linear(hidden_dim, edge_dim)

    def forward(self, x, z):
        x = self.node_input_linear(x).transpose(0, 1)
        x_in = x[:,None,:,:] + x[:,:,None,:]
        z = self.edge_input_linear(z) + x_in
        z_residual = z
        z = self.layer_norm_1(z)
        z = self.mlp(z)
        z = self.layer_norm_2(z + z_residual)
        z = self.edge_out_linear(z)
        return z


def _get_edge_update_params(cls, params):
    cls_args = list(inspect.signature(cls.__init__).parameters.keys())
    edge_params = {}
    for cls_arg in cls_args:
        if cls_arg in params:
            edge_params[cls_arg] = params[cls_arg]
    return edge_params


def get_edge_updater(embed_dim, edge_dim, edge_update_mode, edge_update_params,
                     activation):
    if edge_update_mode is None:
        edge_updater = lambda x, z: z
    elif edge_update_mode in ("framediff", "framediff_add"):
        outer_operation = "concat" if edge_update_mode == "framediff" else "add"
        edge_updater = EdgeUpdaterFrameDiff(
            node_dim=embed_dim,
            edge_dim=edge_dim,
            outer_operation=outer_operation,
            activation=activation
        )
    elif edge_update_mode == "sam_0":
        edge_updater = EdgeUpdaterSAM_0(
            node_dim=embed_dim,
            edge_dim=edge_dim,
            activation=activation,
            **_get_edge_update_params(EdgeUpdaterSAM_0, edge_update_params)
        )
    # elif edge_update_mode == "esmfold":
    #     edge_updater = EdgeUpdaterESMFold(
    #         node_dim=embed_dim,
    #         edge_dim=edge_dim,
    #         activation=activation,
    #         **_get_edge_update_params(EdgeUpdaterESMFold, edge_update_params)
    #     )
    else:
        raise KeyError(edge_update_mode)
    return edge_updater


################################################################################
# Epsilon network.                                                             #
################################################################################

class EpsTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int = 16,
        num_layers: int = 16,
        attention_type: str = "transformer",
        embed_dim: int = 256,
        d_model: int = None,
        num_heads: int = 20,
        mlp_dim: int = None,
        dropout: float = None,
        norm_eps: float = 1e-5,
        norm_pos: str = "pre",
        activation: Union[str, Callable] = "relu",
        out_mode: str = "idpgan",
        time_embed_dim: int = 256,
        time_freq_dim: int = 256,
        use_bead_embed: bool = True,
        bead_embed_dim: int = None,
        use_single_bead: bool = False,
        pt_embed_bead_dim: int = None,
        pos_embed_dim: int = 64,
        use_bias_2d: bool = True,
        pos_embed_r: int = 32,
        use_res_ids: bool = False,
        edge_embed_dim: int = 256,
        edge_embed_mode: str = None,
        edge_update_mode: str = None,
        edge_update_params: dict = {},
        edge_update_freq: int = 1,
        embed_inject_mode: str = "adanorm",
        input_inject_mode: str = None,
        input_inject_pos: str = "out",
        # tem_inject_mode: str = None,
        # tem_inject_pos: str = "out",
        ):
        """
        `input_dim`: dimension of the structural encoding vectors.
        `num_layers`: number of transformer-like layers.
        `attention_type`: type of attention in the attention layers.
        `embed_dim`: dimension of the hidden embeddings of the transformer.
        `d_model`: TODO.
        `num_heads`: number of heads for multi-head attention.
        `mlp_dim`: dimension of the hidden layers of the MLP in transformer
            blocks.
        `dropout`: TODO.
        `norm_eps`: epsilon used in the normalization layers.
        `norm_pos`: position of the normalization layers in the transformer
            blocks. Choices are: 'pre', 'post'.
        `activation`: name of the activation function to use.
        `out_mode`: type of output modules. Choices are: 'idpgan', 'esm'.
        `time_embed_dim`: dimension of the timestep embeddings.
        `time_freq_dim`: TODO.
        `use_bead_embed`: use embedding for beads types (amino acids). Using
            'False' will result in an amino acid unconditioned model.
        `bead_embed_dim`: dimension of the embedding for beads. If 'None' it
            will be the same as `time_embed_dim`. Takes effect only when
            'use_bead_embed' is 'True'.
        `use_single_bead`: TODO.
        `pt_embed_bead_dim`: if providing pre-trained amino acid embeddings
            (like the ESM-2 ones), this argument specifies their dimension.
            The pre-trained embeddings will be projected to `bead_embed_dim`.
            Takes effect only when 'use_bead_embed' is 'True'.
        `pos_embed_dim`: dimension of the positional embeddings.
        `use_bias_2d`: use bias in the linear layers projecting 2d embeddings.
        `pos_embed_r`: threshold seequence separation for 'af2' positional
            embeddings.
        `use_res_ids`: TODO.
        `edge_embed_dim`: TODO.
        `edge_embed_mode`: TODO.
        `edge_update_mode`: TODO.
        `edge_update_params`: TODO.
        `embed_inject_mode`: mechanism for injecting timestep and bead-type
            embeddings in the transformer blocks.
        `input_inject_mode`: mechanism for injecting the intial input of the
            network in its transformer layers. If 'None' the input will not be
            injected at all.
        `input_inject_pos`: position of the module for injecting the input in
            the transformer blocks.
        `tem_inject_mode`: mechanism for injecting a template structure in the
            transformer layers. If 'None' no template will be used at all.
        `tem_inject_pos`: position of the module for injecting the template
            information in the transformer blocks.
        """

        super().__init__()

        ### Check and store the attributes.
        self.use_res_ids = use_res_ids
        self.use_bead_embed = use_bead_embed
        self.use_single_bead = use_single_bead
        self.tem_inject_mode = None

        ### Process input.
        self.project_input = nn.Linear(input_dim, embed_dim)

        ### Shared functions.
        if isinstance(activation, str):
            act_cls = get_act_fn(activation_name=activation)
        else:
            raise TypeError(activation.__class__)

        ### Amino acid embedding.
        if use_bead_embed:
            if bead_embed_dim is None:
                bead_embed_dim = time_embed_dim
            
            self.use_pt_aa_embeddings = pt_embed_bead_dim is not None
            if not self.use_pt_aa_embeddings:  # One-hot encoding.
                self.beads_embed = nn.Embedding(
                    20 if not use_single_bead else 1,
                    bead_embed_dim,
                    # padding_idx=self.padding_idx,
                    )
            else:  # Use some pre-trained embeddings.
                # self.beads_embed = nn.Linear(pt_embed_bead_dim, bead_embed_dim)
                raise NotImplementedError(
                    "Not configured to use pre-trained embeddings.")

        ### Time step embeddings.
        # From the Huggingface DDPM codebase.
        self.time_embed = TimestepEmbedder(
            hidden_size=time_embed_dim,
            frequency_embedding_size=time_freq_dim,
            # activation=act_cls,  # TODO: fix it.
        )

        ### Positional embeddings.
        self.embed_pos = AF2_PositionalEmbedding(
            pos_embed_dim=pos_embed_dim,
            pos_embed_r=pos_embed_r)

        ### Edge embeddings.
        if edge_embed_mode is not None:
            self.embed_edge = EPS_EdgeEmbedder(
                mode=edge_embed_mode,
                edge_dim=edge_embed_dim,
                pos_embed_dim=pos_embed_dim,
                time_embed_dim=time_embed_dim,
                bead_embed_dim=bead_embed_dim if use_bead_embed else None,
                activation=act_cls
            )
        self.edge_embed_mode = edge_embed_mode

        ### Transformer layers.
        if mlp_dim is None:
            mlp_dim = 4 * embed_dim
        
        if edge_embed_mode is None:
            edge_in_dim = pos_embed_dim 
        else:
            edge_in_dim = edge_embed_dim
        
        self.layers = []
        for l in range(num_layers):
            if edge_update_mode is not None:
                # Last layer updater will not be used to compute the output.
                if l < num_layers - 1:
                    if l % edge_update_freq == 0:
                        edge_update_mode_l = edge_update_mode
                    else:
                        edge_update_mode_l = None
                else:
                    edge_update_mode_l = None
            else:
                edge_update_mode_l = None

            layer_l = IdpGAN_TransformerBlock(
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                norm_pos=norm_pos,
                activation=act_cls,
                d_model=d_model,
                add_bias_kv=False,
                attention_type=attention_type,
                embed_inject_mode=embed_inject_mode,
                bead_embed_dim=bead_embed_dim,
                time_embed_dim=time_embed_dim,
                pos_embed_dim=edge_in_dim,
                edge_update_mode=edge_update_mode_l,
                edge_update_params=edge_update_params,
                use_bias_2d=use_bias_2d,
                input_dim=input_dim,
                input_inject_mode=input_inject_mode,
                input_inject_pos=input_inject_pos,
                # tem_inject_mode=tem_inject_mode,
                # tem_inject_pos=tem_inject_pos
            )
            
            self.layers.append(layer_l)
            
        self.layers = nn.ModuleList(self.layers)

        ### Output module.
        if out_mode == "idpgan":
            self.out_module = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                            act_cls(),
                                            nn.Linear(embed_dim, input_dim))
        elif out_mode == "esm":
            self.out_module = nn.Sequential(# nn.LayerNorm(),
                                            nn.Linear(embed_dim, embed_dim),
                                            act_cls(),
                                            nn.LayerNorm(embed_dim),
                                            nn.Linear(embed_dim, input_dim))
        else:
            raise KeyError(out_mode)


    def forward(self, z_t, t, a=None, r=None, z_tem=None):
        """
        z_t: input tensor with shape (B, L, E).
        t: timestep values with shape (B, ).
        a: amino acid tensor with shape (B, L) (if `pt_embed_bead_dim` was
            'None' when initializing the net) or (B, L, pt_embed_bead_dim).
            If 'use_bead_embed' is set to 'False', it should be 'None' instead.
        z_tem: template tensor with shape (B, L, E). Optional, only used if
            `tem_inject_mode` was not 'None' when initializing the net.
        """

        ### Input.
        z_in = z_t.transpose(0, 1)
            
        if z_tem is not None:
            z_tem = z_tem.transpose(0, 1)
        h = self.project_input(z_t)
        h = h.transpose(0, 1)  # (B, L, E) => (L, B, E)

        ### Bead and time embeddings.
        if self.use_bead_embed:
            a_e = self.beads_embed(a).transpose(0, 1)  # (B, L, E_a)
        else:
            a_e = None
        t_e = self.time_embed(t)
        t_e = t_e.unsqueeze(1).repeat(1, h.shape[0], 1).transpose(0, 1)  # (B, L, E_t)

        ### Positional embeddings.
        p = self.embed_pos(h, r=r)
        
        ### Edge embeddings.
        if self.edge_embed_mode is not None:
            p = self.embed_edge(p=p, t=t_e, a=a_e)

        ### Go through all the transformer blocks.
        for layer_idx, layer in enumerate(self.layers):
            h, p = layer(x=h, t=t_e, p=p, a=a_e,
                         x_0=z_in, x_tem=z_tem)

        ### Output module.
        h = self.out_module(h)
        h = h.transpose(0, 1)  # (L, B, E) => (B, L, E)
        eps = h

        return eps


def prepare_eps_input(model, batch):
    if model.net.use_bead_embed:
        if not model.net.use_pt_aa_embeddings:
            a = batch.a
        else:
            a = batch.ae
    else:
        a = None
    if model.net.tem_inject_mode is None:
        x_tem = None
    else:
        x_tem = batch.z_t
    if not model.net.use_res_ids:
        r = None
    else:
        r = batch.r
    return a, r, x_tem

class SAM_EpsTransformer(nn.Module):
    """Wrapper for training and inference experiments."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = EpsTransformer(*args, **kwargs)

    def forward(self, xt, t, batch, num_nodes=None):
        a, r, x_tem = prepare_eps_input(self, batch)
        eps = self.net.forward(z_t=xt, t=t, a=a, r=r, z_tem=x_tem)
        return eps


def get_eps_network(model_cfg):
    # Get the arguments of the eps network class.
    eps_args = list(
        inspect.signature(EpsTransformer.__init__).parameters.keys())
    eps_args.remove("input_dim")
    # Get from 'model_cfg' the corresponding arguments.
    eps_params = {}
    for eps_arg in eps_args:
        if eps_arg in model_cfg["latent_network"]:
            eps_params[eps_arg] = model_cfg["latent_network"][eps_arg]
    # Initialize the network.
    return SAM_EpsTransformer(
        input_dim=model_cfg["generative_model"]["encoding_dim"],
        **eps_params)


if __name__ == "__main__":

    torch.manual_seed(0)

    # Batch size.
    N = 128
    # Number of residues (sequence length).
    L = 12
    # Encoding dimension.
    e_dim = 16

    # Encoding sequence.
    z = torch.randn(N, L, e_dim)

    # Timestep integer values.
    t = torch.randint(0, 1000, (N, ))
    # One-hot encoding for amino acid.
    a = torch.randint(0, 20, (N, L))

    net = EpsTransformer(
        input_dim=e_dim,
        num_layers=2,
        attention_type="transformer",
        embed_dim=256,
        d_model=None,
        num_heads=16,
        mlp_dim=512,
        dropout=None,
        norm_eps=1e-5,
        norm_pos="pre",
        activation="gelu",
        out_mode="idpgan",
        time_embed_dim=256,
        time_freq_dim=256,
        use_bead_embed=True,
        bead_embed_dim=32,
        pt_embed_bead_dim=None,
        pos_embed_dim=64,
        use_bias_2d=True,
        pos_embed_r=32,
        edge_embed_mode="concat",
        edge_embed_dim=128,
        edge_update_mode="framediff",
        edge_update_params={},
        edge_update_freq=4,
        embed_inject_mode="adanorm",
        input_inject_mode="add"
    )
    out = net(z_t=z, t=t, a=a)

    print(out.shape)
    print(out[0])
