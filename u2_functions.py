import numpy as np
import torch
from uncertainties import ufloat

def torch_to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else: 
        return x
    
def coeff_to_u2(coeffs: torch.Tensor) -> torch.Tensor:
    pauli = [torch.tensor([[0.+0.j, 1.+0.j],
                                  [1.+0.j, 0.+0.j]], device=coeffs.device),
             torch.tensor([[0.+0.j, -1.j],
                          [1.j,    0.+0.j]], device=coeffs.device),
             torch.tensor([[1.+0.j,  0.+0.j],
                          [0.+0.j, -1.+0.j]], device=coeffs.device),
             torch.tensor([[1.+0.j,  0.+0.j],
                          [0.+0.j,  1.+0.j]], device=coeffs.device)]
    
    generators = torch.stack(pauli, dim=0) / 2.
    
    a = torch.einsum('iab,...i->...ab', generators, coeffs)
    return fast_matrix_exp(1.j * a)

def u2_to_coeff(U: torch.Tensor) -> torch.Tensor:
    pauli = [torch.tensor([[0.+0.j, 1.+0.j],
                                  [1.+0.j, 0.+0.j]], device=U.device),
             torch.tensor([[0.+0.j, -1.j],
                          [1.j,    0.+0.j]], device=U.device),
             torch.tensor([[1.+0.j,  0.+0.j],
                          [0.+0.j, -1.+0.j]], device=U.device),
             torch.tensor([[1.+0.j,  0.+0.j],
                          [0.+0.j,  1.+0.j]], device=U.device)]
    
    generators = torch.stack(pauli, dim=0) / 2.
    
    U_matrix = -1.j * fast_matrix_log(U)
    U_gen_matrix = torch.einsum('iab,...bc->...iac',generators, U_matrix)
    return 2 * (U_gen_matrix[...,0,0] + U_gen_matrix[...,1,1]).real
    
##### Batched 2x2 matrix operations #####

def fast_matrix_exp(batched_matrix: torch.Tensor) -> torch.Tensor:
    ''''
    Batched matrix: arbitrary number of dimensions, 
    exponential is taken over the the last two, which have to be 2x2
    '''
    batch_shape = batched_matrix.shape
    
    batched_matrix += 1e-8 # exactly zero causes nan
    
    if not batch_shape[-1]==2 and batch_shape[-2]==2:
        raise ValueError(f'Requires 2x2 matrices in the last two dimensions, got {batch_shape}')
    if not (batched_matrix.dtype==torch.cfloat or batched_matrix.dtype==torch.complex128):
        raise TypeError(f'Requires complex torch tensors as input, got {batched_matrix.dtype}')
    
    tau = (batched_matrix[...,0,0] + batched_matrix[...,1,1]) / 2.
    delta = torch.sqrt(((batched_matrix[...,0,0] - batched_matrix[...,1,1]) / 2.)**2 +
                      batched_matrix[...,0,1] * batched_matrix[...,1,0])
    eyes = torch.eye(2, dtype=batched_matrix.dtype)
    eyes = eyes.repeat(*batch_shape[:-2], 1, 1)
    cosh_term = torch.cosh(delta)[...,None,None] * eyes
    sinh_term = torch.sinh(delta)[...,None,None] / delta[...,None,None] *(batched_matrix - tau[...,None,None] * eyes)
    
    exponential = torch.exp(tau)[...,None,None] * (cosh_term + sinh_term) 
    return exponential

def fast_2_eigvals(A: torch.Tensor) -> torch.Tensor:
    # Calculate the Eigenvalues of a batch of 2x2 matrices
    if not A.shape[-1]==2 and A.shape[-2]==2:
        raise ValueError(f'Requires 2x2 matrices in the last two dimensions, got {A.shape}')
    
    A = A.to(torch.cfloat)
    
    trace = A[...,0,0] + A[...,1,1]
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
    return (trace/2. + torch.sqrt(trace**2/4. - det),
            trace/2. - torch.sqrt(trace**2/4. - det))

def fast_matrix_log(A: torch.Tensor, eps=1e-6) -> torch.Tensor:
    ''''
    Batched matrix: arbitrary number of dimensions, 
    log is taken over the the last two, which have to be 2x2
    '''
    trace = A[...,0,0] + A[...,1,1]
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
    
    eig_1, eig_2 = fast_2_eigvals(A)
    eig_1 = eig_1[...,None,None]
    eig_2 = eig_2[...,None,None]

    eye = torch.eye(2, dtype=A.dtype, device=A.device) # shape [2,2]
    eye_expanded = eye.expand(A.shape)
    
    logA = ((A - eye * eig_2) / (eig_1 - eig_2) * torch.log(eig_1) + 
            (A - eye * eig_1) / (eig_2 - eig_1) * torch.log(eig_2))
    
    mask = (torch.abs(eig_1-eig_2) < 1e-7)[...,0,0]
    logA[mask] = torch.eye(2,dtype=A.dtype)[None,None,None,None] * torch.log(eig_1[mask])
    
    return logA

def matmul_2x2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[-2:] == (2, 2) and B.shape[-2:] == (2, 2), "Both matrices must be 2x2"

    C = torch.zeros(A.shape, dtype=A.dtype, device=A.device)

    C[..., 0, 0] = A[..., 0, 0] * B[..., 0, 0] + A[..., 0, 1] * B[..., 1, 0]
    C[..., 0, 1] = A[..., 0, 0] * B[..., 0, 1] + A[..., 0, 1] * B[..., 1, 1]
    C[..., 1, 0] = A[..., 1, 0] * B[..., 0, 0] + A[..., 1, 1] * B[..., 1, 0]
    C[..., 1, 1] = A[..., 1, 0] * B[..., 0, 1] + A[..., 1, 1] * B[..., 1, 1]

    return C

##### U2 observables #####

def bootstrap(function, data, n_resample=1000, return_observable_batch=False):
    n = data.shape[0]
    pos = np.arange(n)
    observable_batch = np.zeros(n_resample)
    for i in range(n_resample):
        indices = np.random.choice(pos, n)
        data_resampled = data[indices]
        observable_batch[i] = torch_to_np(function(data_resampled))
        
    mean = np.mean(observable_batch)
    std = np.std(observable_batch)
    
    if return_observable_batch:
        return mean, std, observable_batch
    else:
        return mean, std
    
def bootstrap_mean(obs):
    return obs.mean()

def wilson_loop(U: torch.Tensor, m: int, n: int) -> torch.Tensor:
    Ux = U[..., 0, :, :]
    Uy = U[..., 1, :, :]
    
    loop = U[..., 0, :, :]
    
    for i in range(m-1):
        loop = matmul_2x2(loop, torch.roll(Ux, shifts=-(i+1), dims=1))
    
    for i in range(n):
        Uy_shifted = torch.roll(Uy, shifts=-m, dims=1)
        loop = matmul_2x2(loop, torch.roll(Uy_shifted, shifts=-(i), dims=2))
                        
    for i in range(m):
        Ux_shifted = torch.roll(Ux, shifts=-n, dims=2)
        Ux_shifted_dagger = Ux_shifted.conj().transpose(-2,-1)
        loop = matmul_2x2(loop, torch.roll(Ux_shifted_dagger, shifts=-(m-i-1), dims=1))
        
    for i in range(n):
        Uy_dagger = Uy.conj().transpose(-2,-1)
        loop = matmul_2x2(loop, torch.roll(Uy_dagger, shifts=-(n-i-1), dims=2))
    
    return loop

def plaquette(U: torch.Tensor) -> torch.Tensor:
    return wilson_loop(U,1,1)

def avg_wilson_loop(U: torch.Tensor, m: int, n: int) -> torch.Tensor:
    loop = wilson_loop(U,m,n)
    loop_trace = loop[...,0,0] + loop[...,1,1]
    return torch.mean(loop_trace.real, dim=(1,2)) / 2. # No clue why but /2 here

def avg_plaquette(U: torch.Tensor) -> torch.Tensor:
    return avg_wilson_loop(U,1,1)

def avg_adjoint_plaquette(U: torch.Tensor) -> torch.Tensor:
    plaq = wilson_loop(U,1,1)
    plaq_trace = plaq[...,0,0] + plaq[...,1,1]
    return torch.mean(((plaq_trace.abs())**2), dim=(1,2)).real # - 1.

def avg_sym_plaquette(U: torch.Tensor) -> torch.Tensor:
    plaq = wilson_loop(U,1,1)
    plaq_trace = plaq[...,0,0] + plaq[...,1,1]
    plaq_squared_trace = torch.einsum('...ij,...ji->...', plaq, plaq)
    return 0.5 * (torch.mean((plaq_trace**2), dim=(1,2)) + torch.mean(plaq_squared_trace, dim=(1,2))).real

def avg_antisym_plaquette(U: torch.Tensor) -> torch.Tensor:
    plaq = wilson_loop(U,1,1)
    plaq_trace = plaq[...,0,0] + plaq[...,1,1]
    plaq_squared_trace = torch.einsum('...ij,...ji->...', plaq, plaq)
    return 0.5 * (torch.mean((plaq_trace**2), dim=(1,2)) - torch.mean(plaq_squared_trace, dim=(1,2))).real

def wilson_action(U: torch.Tensor, beta: float) -> torch.Tensor:
    plaq = plaquette(U)
    
    unit_matrix = torch.zeros_like(plaq)
    unit_matrix[...,0,0], unit_matrix[...,1,1] = 1., 1.
    
    one_minus_plaq = unit_matrix - plaq
    action = (one_minus_plaq[...,0,0] + one_minus_plaq[...,1,1]).real
    
    return beta / 2. * torch.sum(action, dim=(1,2))

def compute_staples(U: torch.Tensor) -> torch.Tensor:
    Ux = U[..., 0, :, :]
    Uy = U[..., 1, :, :]

    Ux_dagger = Ux.conj().transpose(-2,-1)
    Uy_dagger = Uy.conj().transpose(-2,-1)

    direction_0_term_1 = matmul_2x2(torch.roll(Uy, shifts=-1, dims=1), 
                                      torch.roll(Ux_dagger, shifts=-1, dims=2))
    direction_0_term_1 = matmul_2x2(direction_0_term_1, Uy_dagger)

    direction_0_term_2 = matmul_2x2(torch.roll(Uy_dagger, shifts=(-1,1), dims=(1,2)),
                                      torch.roll(Ux_dagger, shifts=1, dims=2))
    direction_0_term_2 = matmul_2x2(direction_0_term_2, torch.roll(Uy, shifts=1, dims=2))
    
    direction_0 = direction_0_term_1 + direction_0_term_2

    direction_1_term_1 = matmul_2x2(torch.roll(Ux_dagger, shifts=(1,-1), dims=(1,2)), 
                                      torch.roll(Uy_dagger, shifts=1, dims=1))
    direction_1_term_1 = matmul_2x2(direction_1_term_1, torch.roll(Ux, shifts=1, dims=1))

    direction_1_term_2 = matmul_2x2(torch.roll(Ux, shifts=-1, dims=2), torch.roll(Uy_dagger, shifts=-1, dims=1))
    direction_1_term_2 = matmul_2x2(direction_1_term_2, Ux_dagger)

    direction_1 = direction_1_term_1 + direction_1_term_2

    staples = torch.stack((direction_0,direction_1), dim=3)
    return staples

def wilson_action_grad(U: torch.Tensor, beta: float) -> torch.Tensor:
    V = compute_staples(U)
    grad = (matmul_2x2(U, V) - 
            matmul_2x2(V.conj().transpose(-2,-1), U.conj().transpose(-2,-1)))
    return -0.125 * beta * 1j * grad
    #return - 1.j * 1./4. * beta * grad
    

def topological_charge(U: torch.Tensor) -> torch.Tensor:
    plaq = plaquette(U)
    theta = torch.angle(torch.det(plaq))
    return torch.sum(theta, dim=(1,2)) / (2. * np.pi)

def topological_susceptibility(U: torch.Tensor):
    Q_hmc = topological_charge(U)
    return (Q_hmc**2).mean() / (U.shape[1]*U.shape[2]) # 256. - normalized by number of plaquettes

def topological_susceptibility_bootstrap(topo_charge):
    return (topo_charge**2).mean() # / 256., moved to calc_observables - normalized by number of plaquettes

def calc_observables(U: torch.Tensor, loop_size=[0,0], give_q_avg=False, n_resample=10000):
    w11_m, w11_std = bootstrap(bootstrap_mean, avg_plaquette(U), n_resample=n_resample)
    topo_susc_m, topo_susc_std = bootstrap(topological_susceptibility_bootstrap, 
                                           topological_charge(U) / np.sqrt(U.shape[1]*U.shape[2]), 
                                           n_resample=n_resample)
    if give_q_avg:
        q_avg_mean, q_avg_std = bootstrap(bootstrap_mean, topological_charge(U), n_resample=n_resample)
        
    if loop_size[0]>1 or loop_size[1]>1:
        wmn = avg_wilson_loop(U, loop_size[0], loop_size[1])
        if give_q_avg:
            return [w11_m, w11_std], [topo_susc_m, topo_susc_std], wmn, [q_avg_mean, q_avg_std]
        else:
            return [w11_m, w11_std], [topo_susc_m, topo_susc_std], wmn
    else:
        if give_q_avg:
            return [w11_m, w11_std], [topo_susc_m, topo_susc_std], [q_avg_mean, q_avg_std]
        else:
            return [w11_m, w11_std], [topo_susc_m, topo_susc_std]
    
def show_results(U: torch.Tensor, loop_size=[0,0], U_compare=None, plots=False):
    if loop_size[0]>1 or loop_size[1]>1:
        plaq, topo_susc, wmn = calc_observables(U, loop_size=loop_size)
    else:
        plaq, topo_susc = calc_observables(U, loop_size=loop_size)
    
    plaq_ufloat = ufloat(plaq[0], plaq[1])
    plaq_topo_susc = ufloat(topo_susc[0], topo_susc[1])

    print('Average plaquette: {:+.1uS}'.format(plaq_ufloat))
    print('Topological susceptibility: {:+.1uS}'.format(plaq_topo_susc))

def calc_wilson_loop_results(U: torch.Tensor,  loop_sizes: list[int], n_resample=10000):
    # loop_sizes: list of loop definitions, e.g. [[2,2],[3,4]]
    results = []
    for loop in loop_sizes:
        w_m, w_std = bootstrap(bootstrap_mean, avg_wilson_loop(U, loop[0], loop[1]), n_resample=n_resample)
        results.append([w_m, w_std])
    return results

def project_to_unitarity(U: torch.Tensor) -> torch.Tensor:
    a = U[:,:,:,:,0,:].clone()
    b = U[:,:,:,:,1,:].clone()

    a = a / torch.sqrt(torch.norm(a,dim=-1)**2)[:,:,:,:,None]
    ab = torch.sum(a.conj()*b,dim=-1)

    b = b - ab[:,:,:,:,None]*a

    b = b / torch.sqrt(torch.norm(b,dim=-1)**2)[:,:,:,:,None]

    U_improved = torch.zeros_like(U)
    U_improved[:,:,:,:,0,:]=a
    U_improved[:,:,:,:,1,:]=b

    return U_improved

n_c = 2

def trace(U: torch.Tensor) -> torch.Tensor:
    return U[...,0,0] + U[...,1,1]

def dagger(U: torch.Tensor) -> torch.Tensor:
    return U.conj().transpose(-2,-1)