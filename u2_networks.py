import os
import numpy as np
import torch
import torch.nn as nn

from u2_functions import plaquette, u2_to_coeff
    
# Bilinear convolution layer

def repack_x(u, w):
    x = torch.cat((u, w), dim=2)
    return x

def matmul_2x2_transport(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[-2:] == (2, 2) and B.shape[-2:] == (2, 2), "Both matrices must be 2x2"

    C = torch.zeros(B.shape, dtype=B.dtype, device=B.device)

    C[..., 0, 0] = A[..., None, 0, 0] * B[..., 0, 0] + A[..., None, 0, 1] * B[..., 1, 0]
    C[..., 0, 1] = A[..., None, 0, 0] * B[..., 0, 1] + A[..., None, 0, 1] * B[..., 1, 1]
    C[..., 1, 0] = A[..., None, 1, 0] * B[..., 0, 0] + A[..., None, 1, 1] * B[..., 1, 0]
    C[..., 1, 1] = A[..., None, 1, 0] * B[..., 0, 1] + A[..., None, 1, 1] * B[..., 1, 1]

    return C

def matmul_2x2_transport_2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[-2:] == (2, 2) and B.shape[-2:] == (2, 2), "Both matrices must be 2x2"

    C = torch.zeros(A.shape, dtype=B.dtype, device=A.device)

    C[..., 0, 0] = A[..., 0, 0] * B[..., None, 0, 0] + A[..., 0, 1] * B[..., None, 1, 0]
    C[..., 0, 1] = A[..., 0, 0] * B[..., None, 0, 1] + A[..., 0, 1] * B[..., None, 1, 1]
    C[..., 1, 0] = A[..., 1, 0] * B[..., None, 0, 0] + A[..., 1, 1] * B[..., None, 1, 0]
    C[..., 1, 1] = A[..., 1, 0] * B[..., None, 0, 1] + A[..., 1, 1] * B[..., None, 1, 1]

    return C

def shift(a, axis, orientation, dims):
    # tensor layout size: [B, N^D, ..., N_C, N_C, 2]
    # add +1 to axis to skip batch dimension

    # old and new shapes for torch.roll
    o_s = tuple(a.shape)
    n_s = (o_s[0], *dims, *tuple(a.shape[2:]))
    a_shift = torch.roll(a.view(*n_s), orientation, axis + 1).view(*o_s)

    return a_shift

def transport(u, w, axis, orientation, dims, use_einsum_alternative=True):
    ws = shift(w, axis, -orientation, dims)

    # select links of appropriate axis
    ua = u.select(dim=2, index=axis)

    if orientation > 0:
        # U W U^dagger
        # A (3,3), B (3,3)
        # C = AB -> ij jk 
        # A (n, m, 3, 3), B(n, m, 3, 3)
        #  
        if use_einsum_alternative:
            wt = matmul_2x2_transport(ua, ws)
            wt = matmul_2x2_transport_2(wt, ua.conj().transpose(-2,-1))
        else:
            wt = torch.einsum('bxij,bxwjk -> bxwik', ua, ws)
            wt = torch.einsum('bxwij,bxkj -> bxwik', wt, ua.conj())
    else:
        # apply shift
        ua = shift(ua, axis, +1, dims)

        if use_einsum_alternative:
            wt = matmul_2x2_transport(ua.conj().transpose(-2,-1), ws)
            wt = matmul_2x2_transport_2(wt, ua)
        else:
            wt = torch.einsum('bxji,bxwjk -> bxwik', ua.conj(), ws)
            wt = torch.einsum('bxwij,bxjk -> bxwik', wt, ua)
    return wt

class LConvBilin(torch.nn.Module):
    """
        A module that merges the LConv and LBilin operation using a simplified parametrization
    """
    def __init__(self, dims, kernel_size, dilation, n_in, n_out, nc, init_w=1.0, use_unit_elements=True, use_symmetric=True):
        super(LConvBilin, self).__init__()
        self.dims = dims
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.nc = nc
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements
        self.use_symmetric = use_symmetric

        # determine kernel structure from `kernel_size` argument
        # example 1: kernel_size=2, symmetric=True -> kernel_range = [-1, 0, +1] * DIM
        # example 2: kernel_size=2, symmetric=False -> kernel_range = [0, +1] * DIM
        # example 3: kernel_size=[-1, +1] -> kernel_range = [-1, 0, +1] * DIM
        self.kernel_range = None
        if isinstance(kernel_size, int):
            if kernel_size > 0:
                if not use_symmetric:
                    self.kernel_range = [[0, kernel_size-1]] * len(dims)
                else:
                    self.kernel_range = [[-(kernel_size-1), kernel_size-1]] * len(dims)
            else:
                raise ValueError("kernel_size should be a positive integer. Got {}.".format(kernel_size))
        elif isinstance(kernel_size, list):
            if len(kernel_size) == 2:
                a, b = kernel_size[0], kernel_size[1]
                if isinstance(a, int) and isinstance(b, int) and a <= 0 and b >= 0:
                    self.kernel_range = [[a, b]] * len(dims)
                else:
                    raise ValueError("kernel_size should be a list containing two integers a,b with a <= 0 and b >= "
                                     "0. Got {}.".format(kernel_size))
            else:
                raise ValueError("kernel_size should be a list containing two integers a,b with a <= 0 and b >= 0. "
                                 "Got {}.".format(kernel_size))
        else:
            raise ValueError("Invalid kernel_size. Got {}.".format(kernel_size))

        # initialize weights
        w_in_size = self.n_in
        t_w_size = self.n_in * (1 + sum([(abs(x[0]) + abs(x[1])) for x in self.kernel_range]))
        w_out_size = self.n_out

        
        # double channels for complex conjugation
        w_in_size = 2 * w_in_size
        t_w_size = 2 * t_w_size

        if self.use_unit_elements:
            w_in_size += 1
            t_w_size += 1

        variance = 1.0 / (w_in_size * t_w_size)
        self.weight = torch.nn.Parameter(data=torch.Tensor(w_out_size, w_in_size, t_w_size), requires_grad=True)
        torch.nn.init.normal_(self.weight.data, std=init_w * np.sqrt(variance))

        """
        print("*** L-ConvBilin layer")
        print(f"kernel_range: {self.kernel_range}")
        print(f"weight.shape: {self.weight.shape}")
        print(f"Number of weights: {np.prod(self.weight.shape)}")
        """

        # set residual and bias terms to zero
        torch.nn.init.constant_(self.weight.data[:, :, -1], 0.0)
        torch.nn.init.constant_(self.weight.data[:, -1, :], 0.0)

        # construct unit matrix
        self.unit_matrix = torch.eye(self.nc)

        # caches for unit_tensor
        self.unit_tensors = {}

    def get_unit(self, shape, device, dtype):
        shape = tuple(shape)

        if shape not in self.unit_tensors.keys():
            unit_matrix = self.unit_matrix.to(device).to(dtype)
            unit_tensor = unit_matrix.expand(shape)
            self.unit_tensors[shape] = unit_tensor
        
        return self.unit_tensors[shape]


    def forward(self, u, w):
        global bilin_implementation, transport_implementation
        
        # transported terms
        t_w = [w]
        # gather all terms along lattice axes up to kernel_size
        for axis in range(len(self.dims)):
            for i, o in zip([0, 1], [-1, +1]):
                w_transport = w
                kernel_size = abs(self.kernel_range[axis][i])
                for d in range(kernel_size):
                    # get transported terms
                    for step in range(self.dilation):
                        # TODO: use complex implementation here
                        w_transport = transport(u, w_transport, axis=axis, orientation=o, dims=self.dims)
                    # and add to list
                    t_w.append(w_transport)
        # combine terms into a single tensor
        t_w = torch.cat(t_w, dim=2)
        
        # enlarge tensors by complex conjugates
        w = torch.cat((w, torch.transpose(w.conj(), 3, 4).contiguous()), dim=2)
        t_w = torch.cat((t_w, torch.transpose(t_w.conj(), 3, 4).contiguous()), dim=2) 

        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_tensor = self.get_unit(unit_shape, device=w.device, dtype=w.dtype)

            w   = repack_x(w,   unit_tensor)
            t_w = repack_x(t_w, unit_tensor)
        
        # v2: complex multiplication, two einsums, multiply transported Ws first [the good one]
        weight_c = self.weight.type(w.dtype)
        tmp = torch.einsum('uvw, bxwjk -> bxuvjk', weight_c, t_w)
        w = torch.einsum('bxvij, bxuvjk -> bxuik', w, tmp)
        return w

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")

        self.dims = dims

# Time embedding

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, t):
        t_proj = 2 * np.pi * t[:,None] * self.W[None,:]
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]
    
# Full L-CNN

class LCNN(nn.Module):
    def __init__(self, channels=[16], n_layers=2, kernel_sizes=[2,2], log_time_embedding=False, time_embed_dim=256, time_embed_scale=30., shift_time_embedding=0.):
        super().__init__()
        
        self.n_layers = n_layers

        self.log_time_embedding = log_time_embedding
        self.shift_time_embedding = shift_time_embedding
        self.embed = GaussianFourierProjection(embed_dim=time_embed_dim, scale=time_embed_scale)
        
        # Input layer
        self.cb_layers = nn.ModuleList([LConvBilin(dims=[16,16],
                              kernel_size=kernel_sizes[0], 
                              dilation=1, 
                              n_in=1, 
                              n_out=channels[0], 
                              nc=2, 
                              init_w=1.0, 
                              use_unit_elements=True, 
                              use_symmetric=True)])
        
        # Intermediate layers
        for i in range(n_layers-2):
            self.cb_layers.append(LConvBilin(dims=[16,16],
                              kernel_size=kernel_sizes[i+1], 
                              dilation=1, 
                              n_in=channels[i], 
                              n_out=channels[i+1], 
                              nc=2, 
                              init_w=1.0, 
                              use_unit_elements=True, 
                              use_symmetric=True))
        
        # Output layer
        self.cb_layers.append(LConvBilin(dims=[16,16],
                              kernel_size=kernel_sizes[-1], 
                              dilation=1, 
                              n_in=channels[-1], 
                              n_out=2, 
                              nc=2, 
                              init_w=1.0, 
                              use_unit_elements=True, 
                              use_symmetric=True))

        self.dense_layers = nn.ModuleList([Dense(time_embed_dim, channels[0])])
        for i in range(n_layers-2):
            self.dense_layers.append(Dense(time_embed_dim, channels[i+1]))

        self.silu = nn.SiLU()

        # construct unit matrix
        self.unit_matrix = torch.eye(2)

        # caches for unit_tensor
        self.unit_tensors = {}

    def get_unit(self, shape, device, dtype):
        shape = tuple(shape)

        if shape not in self.unit_tensors.keys():
            unit_matrix = self.unit_matrix.to(device).to(dtype)
            unit_tensor = unit_matrix.expand(shape)
            self.unit_tensors[shape] = unit_tensor
        
        return self.unit_tensors[shape]

    def forward(self, u, t):
        t = t + self.shift_time_embedding
        if self.log_time_embedding:
            t = (torch.log(t) - 1e-3) / np.log(1e-3) # The -1e-3 should avoid having t=0 in the embedding layer
        embed = self.silu(self.embed(t))
        
        u_shape = u.shape
        # Calculate plaquettes:
        w = plaquette(u)[:,:,:,None,:,:]
        w_shape = w.shape
        # Put xy in single dimension
        u = u.reshape(u.shape[0], u.shape[1]*u.shape[2], u.shape[3], 2, 2)
        w = w.reshape(w.shape[0], w.shape[1]*w.shape[2], w.shape[3], 2, 2)

        for i in range(self.n_layers-1):
            # Bilin Conv layer
            w = self.cb_layers[i](u, w)
            # Time information
            unit = self.get_unit(w.shape, w.device, w.dtype)
            w += unit * self.dense_layers[i](embed)[:,None,:,:,:]

        # Bilin Conv layer
        w = self.cb_layers[-1](u, w)

        # reshape to x y seperate dims
        w = w.reshape(w_shape[0],w_shape[1],w_shape[2],2,2,2)

        w = self.ah(w)
        return w
    
    def ah(self, U):
        # Calculate the anti-hermitian part / i
        U = 1/(2.j) * (U - U.conj().transpose(-2,-1))
        return U

    def update_dims(self, dims):
        if len(dims) != len(self.cb_layers[0].dims):
            raise ValueError("Length of new 'dims' must be the same as previous 'dims'.")
        for i in range(len(self.cb_layers)):
            self.cb_layers[i].update_dims(dims)