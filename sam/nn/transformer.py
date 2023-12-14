import numpy as np
import torch
import torch.nn as nn


################################################################################
# Custom attention layers.                                                     #
################################################################################

class TransformerLayer(nn.Module):

    def __init__(self, in_dim,
                 d_model, nhead,
                 dp_attn_norm="d_model",
                 in_dim_2d=None,
                 use_bias_2d=True):
        """d_model = c*n_head"""

        super(TransformerLayer, self).__init__()

        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        self.in_dim_2d = in_dim_2d

        if dp_attn_norm not in ("d_model", "head_dim"):
            raise KeyError("Unkown 'dp_attn_norm': %s" % dp_attn_norm)
        self.dp_attn_norm = dp_attn_norm

        # Linear layers for q, k, v for dot product affinities.
        self.q_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.k_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.v_linear = nn.Linear(in_dim, self.d_model, bias=False)

        # Output layer.
        out_linear_in = self.d_model
        self.out_linear = nn.Linear(out_linear_in, in_dim)

        # Branch for 2d representation.
        if self.in_dim_2d is not None:
            self.mlp_2d = nn.Sequential(# nn.Linear(in_dim_2d, in_dim_2d),
                                        # nn.ReLU(),
                                        # nn.Linear(in_dim_2d, in_dim_2d),
                                        # nn.ReLU(),
                                        nn.Linear(in_dim_2d, self.nhead,
                                                  bias=use_bias_2d))


    verbose = False
    def forward(self, s, _k, _v, p):

        #----------------------
        # Prepare the  input. -
        #----------------------

        # Receives a (L, N, I) tensor.
        # L: sequence length,
        # N: batch size,
        # I: input embedding dimension.
        seq_l, b_size, _e_size = s.shape
        if self.dp_attn_norm == "d_model":
            w_t = 1/np.sqrt(self.d_model)
        elif self.dp_attn_norm == "head_dim":
            w_t = 1/np.sqrt(self.head_dim)
        else:
            raise KeyError(self.dp_attn_norm)

        #----------------------------------------------
        # Compute q, k, v for dot product affinities. -
        #----------------------------------------------

        # Compute q, k, v vectors. Will reshape to (L, N, D*H).
        # D: number of dimensions per head,
        # H: number of head,
        # E = D*H: embedding dimension.
        q = self.q_linear(s)
        k = self.k_linear(s)
        v = self.v_linear(s)

        # Actually compute dot prodcut affinities.
        # Reshape first to (N*H, L, D).
        q = q.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        q = q * w_t
        k = k.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)

        # Then perform matrix multiplication between two batches of matrices.
        # (N*H, L, D) x (N*H, D, L) -> (N*H, L, L)
        dp_aff = torch.bmm(q, k.transpose(-2, -1))

        #--------------------------------
        # Compute the attention values. -
        #--------------------------------

        tot_aff = dp_aff

        # Use the 2d branch.
        if self.in_dim_2d is not None:
            p = self.mlp_2d(p)
            # (N, L1, L2, H) -> (N, H, L2, L1)
            p = p.transpose(1, 3)
            # (N, H, L2, L1) -> (N, H, L1, L2)
            p = p.transpose(2, 3)
            # (N, H, L1, L2) -> (N*H, L1, L2)
            p = p.contiguous().view(b_size*self.nhead, seq_l, seq_l)
            tot_aff = tot_aff + p

        attn = nn.functional.softmax(tot_aff, dim=-1)
        # if dropout_p > 0.0:
        #     attn = dropout(attn, p=dropout_p)

        #-----------------
        # Update values. -
        #-----------------

        # Update values obtained in the dot product affinity branch.
        s_new = torch.bmm(attn, v)
        # Reshape the output, that has a shape of (N*H, L, D) back to (L, N, D*H).
        s_new = s_new.transpose(0, 1).contiguous().view(seq_l, b_size, self.d_model)

        # Compute the ouput.
        output = s_new
        output = self.out_linear(output)
        return (output, )


class TransformerTimewarpLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 d_model,
                 nhead,
                 in_dim_2d=None,
                 use_bias_2d=True,
                 v_dim_mode="custom",
                 eps_elle=0):
        """TODO"""

        super(TransformerTimewarpLayer, self).__init__()

        self.nhead = nhead
        self.in_dim_2d = in_dim_2d

        # Linear layers for q, k, v for dot product affinities.
        self.n_points = 1
        self.n_coords = 3
        self.q_linear = nn.Linear(in_dim, self.nhead*self.n_points*self.n_coords,
                                  bias=False)
        self.k_linear = nn.Linear(in_dim, self.nhead*self.n_points*self.n_coords,
                                  bias=False)
        if v_dim_mode == "custom":
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
            self.v_dim_head = d_model//nhead
        else:
            self.v_dim_head = self.n_points*self.n_coords
        self.v_dim = self.nhead*self.v_dim_head
            
        self.v_linear = nn.Linear(in_dim, self.v_dim,
                                  bias=False)

        # elle_init = torch.randn(1, self.nhead, 1, 1))
        elle_init = torch.log(torch.exp(torch.full((1, self.nhead, 1, 1), 1.)) - 1.)
        self.elle = nn.Parameter(elle_init)
        self.eps_elle = eps_elle

        # Output layer.
        self.out_linear = nn.Linear(self.v_dim, in_dim)

        # Branch for 2d representation.
        if self.in_dim_2d is not None:
            self.mlp_2d = nn.Sequential(# nn.Linear(in_dim_2d, in_dim_2d),
                                        # nn.ReLU(),
                                        # nn.Linear(in_dim_2d, in_dim_2d),
                                        # nn.ReLU(),
                                        nn.Linear(in_dim_2d, self.nhead,
                                                  bias=use_bias_2d))


    verbose = False
    def forward(self, q, k, v, p):

        #----------------------
        # Prepare the  input. -
        #----------------------

        # Receives a (L, N, I) tensor.
        # L: sequence length,
        # N: batch size,
        # I: input embedding dimension.
        seq_l, b_size, _e_size = q.shape

        #---------------------------------------
        # Compute squared distance affinities. -
        #---------------------------------------

        # Compute q, k, v vectors. Will reshape to (L, N, 3*H).
        # H: number of head,
        q3 = self.q_linear(q)
        k3 = self.k_linear(k)
        v3 = self.v_linear(v)

        # print("---")
        # print("q3:", q3.shape)
        # print("k3:", k3.shape)
        # print("v3:", v3.shape)

        # Actually compute the square distances.
        # Reshape first to (N*H, L, D).
        q3 = q3.contiguous().view(seq_l, b_size*self.nhead,
                                  self.n_points, self.n_coords).transpose(0, 1)
        k3 = k3.contiguous().view(seq_l, b_size*self.nhead,
                                  self.n_points, self.n_coords).transpose(0, 1)
        v3 = v3.contiguous().view(seq_l, b_size*self.nhead,
                                  self.v_dim_head).transpose(0, 1)
        # print("q3_r:", q3.shape)
        # print("k3_r:", k3.shape)
        # print("v3_r:", v3.shape)

        # Compute the difference between the coordinates.
        sd_aff = q3[:,None,...] - k3[:,:,None,...]
        # print("sd_aff_0:", sd_aff.shape)
        # Compute the squared distances.
        sd_aff = torch.sum(torch.square(sd_aff), axis=-1)
        # print("sd_aff_1:", sd_aff.shape)
        # Sum over points.
        sd_aff = torch.sum(sd_aff, axis=-1)
        # print("sd_aff_2:", sd_aff.shape)
        # Multiply by weights.
        # sd_aff = (w_c/2)*sd_aff
        # Divide by elle.
        sd_aff = sd_aff.view(b_size, self.nhead, seq_l, seq_l)
        # print("sd_aff_3:", sd_aff.shape)
        elle_sp = nn.functional.softplus(self.elle)**2
        # print("elle_sp:", elle_sp.shape)
        sd_aff = sd_aff/(elle_sp+self.eps_elle)
        sd_aff = sd_aff.view(b_size*self.nhead, seq_l, seq_l)
        # print("sd_aff_4:", sd_aff.shape)

        #--------------------------------
        # Compute the attention values. -
        #--------------------------------

        tot_aff = -sd_aff

        # Use the 2d branch.
        if self.in_dim_2d is not None:
            p = self.mlp_2d(p)
            # (N, L1, L2, H) -> (N, H, L2, L1)
            p = p.transpose(1, 3)
            # (N, H, L2, L1) -> (N, H, L1, L2)
            p = p.transpose(2, 3)
            # (N, H, L1, L2) -> (N*H, L1, L2)
            p = p.contiguous().view(b_size*self.nhead, seq_l, seq_l)
            tot_aff = tot_aff + p
        
        attn = nn.functional.softmax(tot_aff, dim=-1)
        # print("attn:", attn.shape)

        #-----------------
        # Update values. -
        #-----------------
        
        # Update values obtained in the squared distances branch.
        s3_new = torch.bmm(attn, v3)
        # Reshape the output, that has a shape of TODO.
        s3_new = s3_new.transpose(0, 1).contiguous().view(
            seq_l, b_size, self.v_dim)
        s_new = self.out_linear(s3_new)
        return (s_new, )


################################################################################
# Pytorch attention layer.                                                     #
################################################################################

class PyTorchAttentionLayer(nn.Module):
    """New module documentation: TODO."""

    def __init__(self, embed_dim, num_heads, edge_dim,
                 add_bias_kv=False, use_bias_2d=True, dropout=0.0):
        """Arguments: TODO."""

        super().__init__()

        # Attributes to store.
        self.num_heads = num_heads

        # Multihead attention.
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False)

        # Project edge features.
        self.edge_to_bias = nn.Linear(edge_dim, num_heads, bias=use_bias_2d)

    def forward(self, q, k, v, p):
        b_size = q.shape[1]
        seq_l = q.shape[0]
        p = self.edge_to_bias(p)
        p = p.transpose(1, 3).transpose(2, 3)
        p = p.contiguous().view(b_size*self.num_heads, seq_l, seq_l)
        out = self.mha(q, k, v, attn_mask=p)
        return out