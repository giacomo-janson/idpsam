from typing import Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Common functions for all neural networks.                                    #
################################################################################

def get_act_fn(activation_name, **kwargs):
    if activation_name == "relu":
        return nn.ReLU
    elif activation_name == "elu":
        return nn.ELU
    elif activation_name == "lrelu":
        return nn.LeakyReLU  # return functools.partial(nn.LeakyReLU, negative_slope=0.1)
    elif activation_name == "prelu":
        return nn.PReLU
    elif activation_name in ("swish", "silu"):
        return nn.SiLU
    elif activation_name == "gelu":
        return nn.GELU
    else:
        raise KeyError(activation_name)

def get_num_params(net):
    num_params = 0
    if hasattr(net, "parameters"):
        for parameter in net.parameters():
            if hasattr(parameter, "numel"):
                num_params += parameter.numel()
    return num_params


################################################################################
# Common modules.                                                              #
################################################################################

class MLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = None,
                 activation: Callable = nn.ReLU,
                 n_hidden_layers: int = 1,
                 # final_activation: bool = False,
                 final_norm: bool = False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else out_dim
        layers = [nn.Linear(in_dim, hidden_dim),
                  activation()]
        for l in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim),
                           activation()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        # if final_activation:
        #     layers.append(activation)
        if final_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# class SmoothConv(nn.Module):
    
#     def __init__(self, probs, n_channels=1):
#         """
#         `probs`: example [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]
#         """
#         super().__init__()
#         # Create kernels.
#         kernel = torch.FloatTensor([[probs]])
#         kernel = kernel.repeat(n_channels, 1, 1)
#         self.register_buffer('kernel', kernel)
#         self.n_channels = n_channels
        
#     def forward(self, x):
#         # Apply smoothing.
#         x = F.conv1d(x, self.kernel, padding="same", groups=self.n_channels)
#         return x


# ################################################################################
# # GNN classes.                                                                 #
# ################################################################################

# class UpdaterModule(nn.Module):

#     def __init__(self, gnn_layer, in_dim,
#                  dim_feedforward, activation,
#                  dropout=0.0,
#                  layer_norm_eps=1e-5):
#         super(UpdaterModule, self).__init__()
#         self.gnn_layer = gnn_layer
#         self.linear1 = nn.Linear(in_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, in_dim)
#         self.use_norm = layer_norm_eps is not None
#         if self.use_norm:
#             self.norm1 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
#             self.norm2 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = activation
#         self.update_module = nn.Sequential(self.linear1,
#                                            self.activation,
#                                            self.linear2)

#     def forward(self, h, edge_index, edge_attr=None):
#         h2 = self.gnn_layer(h, edge_index, edge_attr=edge_attr)
#         h = h + self.dropout1(h2)
#         if self.use_norm:
#             h = self.norm1(h)
#         # s2 = self.linear2(self.dropout(self.activation(self.linear1(s))))
#         h2 = self.update_module(h)
#         h = h + self.dropout2(h2)
#         if self.use_norm:
#             h = self.norm2(h)
#         return h


# class EdgeUpdaterModule(nn.Module):

#     def __init__(self, gnn_layer, in_edge_dim, out_edge_dim, activation):
#         super(EdgeUpdaterModule, self).__init__()
#         self.gnn_layer = gnn_layer
#         self.in_edge_dim = in_edge_dim
#         self.out_edge_dim = out_edge_dim
#         self.activation = activation

#         self.edge_mlp = nn.Sequential(nn.Linear(in_edge_dim, out_edge_dim),
#                                       activation,
#                                       nn.Linear(out_edge_dim, out_edge_dim))

#     def forward(self, h, edge_index, edge_attr):
#         edge_attr = self.edge_mlp(edge_attr)
#         return self.gnn_layer(h, edge_index, edge_attr=edge_attr)


################################################################################
# Positional embeddings.                                                       #
################################################################################

class AF2_PositionalEmbedding(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
                 pos_embed_dim: int,
                 pos_embed_r: int = 32,
                 dim_order: str = "transformer"
                ):
        super().__init__()
        self.embed = nn.Embedding(pos_embed_r*2+1, pos_embed_dim)
        self.pos_embed_r = pos_embed_r
        self.set_dim_order(dim_order)

    def set_dim_order(self, dim_order):
        self.dim_order = dim_order
        if self.dim_order == "transformer":
            self.l_idx = 0  # Token (residue) index.
            self.b_idx = 1  # Batch index.
        elif self.dim_order == "trajectory":
            self.l_idx = 1
            self.b_idx = 0
        else:
            raise KeyError(dim_order)

    def forward(self, x, r=None):
        """
        x: xyz coordinate tensor of shape (L, B, *) if `dim_order` is set to
            'transformer'.
        r: optional, residue indices tensor of shape (B, L).

        returns:
        p: 2d positional embedding of shape (B, L, L, `pos_embed_dim`).
        """
        if r is None:
            prot_l = x.shape[self.l_idx]
            p = torch.arange(0, prot_l, device=x.device)
            p = p[None,:] - p[:,None]
            bins = torch.linspace(-self.pos_embed_r, self.pos_embed_r,
                                self.pos_embed_r*2+1, device=x.device)
            b = torch.argmin(
                torch.abs(bins.view(1, 1, -1) - p.view(p.shape[0], p.shape[1], 1)),
                axis=-1)
            p = self.embed(b)
            p = p.repeat(x.shape[self.b_idx], 1, 1, 1)
        else:
            b = r[:,None,:] - r[:,:,None]
            b = torch.clip(b, min=-self.pos_embed_r, max=self.pos_embed_r)
            b = b + self.pos_embed_r
            p = self.embed(b)
        return p