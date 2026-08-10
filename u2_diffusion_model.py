import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from u2_functions import (torch_to_np, matmul_2x2, fast_matrix_exp, fast_matrix_log, wilson_action, topological_charge, 
                          calc_observables, project_to_unitarity, calc_wilson_loop_results, wilson_action_grad)

class u2_dm():
    def __init__(self, 
                 network, 
                 grid_size, 
                 sde_type, 
                 sde_param, 
                 t_max=1., 
                 t_n=200, 
                 use_log_spaced_t=False, 
                 t_log_min=-3, 
                 t_log_max=0, 
                 use_2x2matmul=True):
        self.network = network
        self.grid_size = grid_size
        self.use_2x2matmul = use_2x2matmul

        pauli = [torch.tensor([[0.+0.j, 1.+0.j],
                              [1.+0.j, 0.+0.j]]),
                 torch.tensor([[0.+0.j, -1.j],
                              [1.j,    0.+0.j]]),
                 torch.tensor([[1.+0.j,  0.+0.j],
                              [0.+0.j, -1.+0.j]]),
                 torch.tensor([[1.+0.j,  0.+0.j],
                              [0.+0.j,  1.+0.j]])]

        self.generators = torch.stack(pauli, dim=0) / 2.
        
        self.t_n = t_n
        if use_log_spaced_t:
            self.t_steps = torch.logspace(t_log_min, t_log_max, t_n-1)
            self.t_steps = torch.cat((torch.tensor([0.]), self.t_steps))
        else:
            self.t_steps = torch.linspace(0., t_max, t_n)

        self.sde_type = sde_type
        self.sde_param = sde_param
        if sde_type=='ve':
            # sde_param...sigma
            self.p0t_std = torch.sqrt((sde_param**(2.*self.t_steps) - 1.) / (2. * np.log(sde_param)))
        elif sde_type=='vp':
            # sde_param...omega
            self.p0t_std = (1 - torch.exp(-2. * sde_param * self.t_steps)) / sde_param
            self.p0t_mean_factor = torch.exp(-sde_param * self.t_steps)
        self.rand_t_val = None
    
    def diffusion_step(self, U, diffusion_coeff, drift_term=0., return_noise=False):
        if diffusion_coeff is not None:
            randn = torch.randn((U.shape[0],self.grid_size,self.grid_size,2,4)).to(torch.cfloat)
            rand_algebra = torch.einsum('iab,...i->...ab', self.generators, randn)

            diffusion_term = diffusion_coeff[:,None,None,None,None,None] * rand_algebra
            perturbation = fast_matrix_exp(1.j * (diffusion_term + drift_term))
        else:
            perturbation = fast_matrix_exp(1.j * (drift_term))
        if return_noise:
            if self.use_2x2matmul:
                return matmul_2x2(perturbation, U), rand_algebra
            else:
                return torch.einsum('...ij,...jk->...ik', perturbation, U), rand_algebra
        else:
            if self.use_2x2matmul:
                return matmul_2x2(perturbation, U)
            else:
                return torch.einsum('...ij,...jk->...ik', perturbation, U)
    
    def loss_function(self, U0, lambda_term=False):
        rand_indices = torch.randint(1,self.t_n,(U0.shape[0],))
        rand_indices.sort()
        rand_t = self.t_steps[rand_indices]
        rand_p0t_std = self.p0t_std[rand_indices]
        if self.sde_type=='vp':
            rand_p0t_mean_factor = self.p0t_mean_factor[rand_indices]

        if self.sde_type=='vp':
            U0_scaled = fast_matrix_exp(fast_matrix_log(U0) * rand_p0t_mean_factor[:,None,None,None,None,None])
            Ut, rand_algebra = self.diffusion_step(U0_scaled, rand_p0t_std, return_noise=True)
        else:
            Ut, rand_algebra = self.diffusion_step(U0, rand_p0t_std, return_noise=True)
        
        score = self.network(Ut, rand_t).to(torch.cfloat)
        
        loss = (score + rand_algebra / rand_p0t_std[:,None,None,None,None,None])
        if lambda_term:
            loss = loss * torch.sqrt(rand_p0t_std[:,None,None,None,None,None])
        loss = torch.einsum('...ij,...ji->...', loss, loss)
        loss = loss.real
        loss = torch.sum(loss, dim=(3))
        return torch.mean(loss)
    
    def calc_validation_loss(self, U0, lambda_term=False):
        if self.rand_t_val is None:
            rand_indices = torch.randint(1,self.t_n,(U0.shape[0],))
            rand_indices.sort()
            self.rand_t_val = self.t_steps[rand_indices]
            self.rand_p0t_std_val = self.p0t_std[rand_indices]
            randn = torch.randn((U0.shape[0],self.grid_size,self.grid_size,2,4)).to(torch.cfloat)
            self.rand_algebra_val = torch.einsum('iab,...i->...ab', self.generators, randn)
            diffusion_term = self.rand_p0t_std_val[:,None,None,None,None,None] * self.rand_algebra_val
            perturbation = fast_matrix_exp(1.j * (diffusion_term))
            self.Ut = matmul_2x2(perturbation, U0)

        #Ut, rand_algebra = self.diffusion_step(U0, self.rand_p0t_std_val, return_noise=True) # Assuming VE scheme

        with torch.no_grad():
            score = self.network(self.Ut, self.rand_t_val).to(torch.cfloat)
        
        loss = (score + self.rand_algebra_val / self.rand_p0t_std_val[:,None,None,None,None,None])
        if lambda_term:
            loss = loss * torch.sqrt(self.rand_p0t_std_val[:,None,None,None,None,None])
        loss = torch.einsum('...ij,...ji->...', loss, loss)
        loss = loss.real
        loss = torch.sum(loss, dim=(3))
        return torch.mean(loss).cpu().numpy()
    
    def train(self, 
              data_loader: DataLoader, 
              n_epochs: int, 
              lr: float, 
              lr_decay: float, 
              lambda_term=False, 
              ckpt_interval=None, 
              ckpt_path=None, 
              progress_bar=False, 
              get_validation_loss=False, 
              validation_set=None):
        if hasattr(self, "optimizer"):
            self.optimizer.param_groups[0]['lr'] = lr
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)

        if get_validation_loss:
            loss_history = []
    
        if progress_bar:
            epochs = tqdm.tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)

        for n in epochs:
            for batch in data_loader:
                loss = self.loss_function(batch[0], lambda_term=lambda_term)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if (any(torch.isnan(param).any() for param in self.network.state_dict().values())):
                raise ValueError(f'Some of the network parameters are nan. Epoch: {n}/{n_epochs}')
            self.scheduler.step()

            if ckpt_interval is not None:
                if n%ckpt_interval==0:
                    torch.save(self.network, ckpt_path+f'network_epoch{n}.pt')

            if get_validation_loss:
                val_loss = self.calc_validation_loss(validation_set, lambda_term=False)
                loss_history.append(val_loss)

        if get_validation_loss:
            return loss_history
            
    def generate_init_config(self, n: int) -> torch.Tensor:
        randn = torch.randn((n, self.grid_size, self.grid_size, 2, 4)).to(torch.cfloat) * self.p0t_std[-1]
        rand_generators = torch.einsum('iab,...i->...ab', self.generators, randn)
        return fast_matrix_exp(1.j * rand_generators)
    
    def metropolis_step(self, 
                        beta: float, 
                        U_old: torch.Tensor, 
                        U_new: torch.Tensor, 
                        score_old: torch.Tensor, 
                        score_new: torch.Tensor, 
                        dtau: float, 
                        get_acc_rate=False):
        action_old = wilson_action(U_old, beta)
        action_new = wilson_action(U_new, beta)

        if self.use_2x2matmul:
            log_unew_u_inv = fast_matrix_log(matmul_2x2(U_new, U_old.conj().transpose(-2,-1)))
        else:
            log_unew_u_inv = fast_matrix_log(torch.einsum("...ij,...jk->...ik", U_new, U_old.conj().transpose(-2,-1)))
        eta_old_to_new = -1.j * log_unew_u_inv - dtau * score_old
        q_old_to_new = torch.einsum("...ij,...ji->...", eta_old_to_new, eta_old_to_new)
        q_old_to_new = 1. / (2*dtau) * torch.sum(q_old_to_new, dim=(1,2,3)).real

        if self.use_2x2matmul:
            log_u_unew_inv = fast_matrix_log(matmul_2x2(U_old, U_new.conj().transpose(-2,-1)))
        else:
            log_u_unew_inv = fast_matrix_log(torch.einsum("...ij,...jk->...ik", U_old, U_new.conj().transpose(-2,-1)))
        eta_new_to_old = -1.j * log_u_unew_inv - dtau * score_new
        q_new_to_old = torch.einsum("...ij,...ji->...", eta_new_to_old, eta_new_to_old)
        q_new_to_old = 1. / (2*dtau) * torch.sum(q_new_to_old, dim=(1,2,3)).real

        accept_prob = torch.exp(action_old - action_new + q_old_to_new - q_new_to_old)

        U = torch.zeros_like(U_old)
        update_check = torch.ge(accept_prob, torch.rand((U_old.shape[0])))

        U[update_check] = U_new[update_check]
        U[~update_check] = U_old[~update_check]

        if get_acc_rate:
            acc_rate = update_check.sum() / len(update_check)
            return U, acc_rate
        else:
            return U
    
    def maala_sampler(self, 
                      n: int, 
                      beta: float, 
                      beta_0: float, 
                      dtau: float, 
                      n_tau: int, 
                      metropolis_threshold: float, 
                      analytic_action_step=False, 
                      return_hist=False, 
                      return_langevin_hist=False, 
                      return_topo_charge_hist=False, 
                      save_intermediate_configs=False, 
                      folder_path=None,
                      progress_bar=False, 
                      n_resample=1000,
                      tau_threshold_list=None, 
                      n_tau_list=None):
        U = self.generate_init_config(n)
        
        acc_rate_hist = {'t': [], 'tau': [], 'acc_rate': []}

        if return_hist:
            observable_hist = {'t': [], 'avg_plaquette': [], 'topo_susc': [], 'q_avg': [], '2x2': [], '3x3': [], "4x4": []}

        if return_langevin_hist:
            t_of_hists = []
            hists = []

        if return_topo_charge_hist:
            t_hist = [1.]
            topo_charge_hist = torch.zeros((1,n))
            topo_charge_hist[0] = topological_charge(U)
        
        if save_intermediate_configs: # This assumes all other hist options are set to False
            config_path = folder_path+'/config_hist/'
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            t_hist = []
        
        if analytic_action_step:
            t_n_steps = self.t_n
        else:
            t_n_steps = self.t_n - 1
           
        if progress_bar:
            epochs = tqdm.tqdm(range(t_n_steps))
        else: 
            epochs = range(t_n_steps)

        for i in epochs:
            t_batch = torch.ones(n) * self.t_steps[-i-1]

            if return_langevin_hist and i%10==0:
                observable_hist = {'tau': [], 'avg_plaquette': [], 'topo_susc': []}
            
            if tau_threshold_list is not None:
                for l in range(len(tau_threshold_list)):
                    if i < tau_threshold_list[l]:
                        n_tau = n_tau_list[l]
                if i > tau_threshold_list[-1]:
                    n_tau = n_tau_list[-1]
            for j in range(n_tau):
                if i < t_n_steps-1:
                    with torch.no_grad():
                        score = beta / beta_0 * self.network(U, t_batch).to(torch.cfloat)
                else: # Final SQ step with analytic action gradient
                    score = - wilson_action_grad(U, beta)
                
                diffusion_factor = np.sqrt(2.*dtau) * torch.ones(n)
                U_new = self.diffusion_step(U, diffusion_factor, drift_term=dtau*score)
                
                if i>=metropolis_threshold*self.t_n:
                    if i < t_n_steps-1:
                        with torch.no_grad():
                            score_new = beta / beta_0 * self.network(U_new, t_batch).to(torch.cfloat)
                    else: # Final SQ step with analytic action gradient
                        score_new = - wilson_action_grad(U_new, beta)

                    U, acc_rate = self.metropolis_step(beta, U, U_new, score, score_new, dtau, get_acc_rate=True)
                    acc_rate_hist['t'].append((torch_to_np(self.t_steps[-i-1])))
                    acc_rate_hist['tau'].append((j+1)*dtau)
                    acc_rate_hist['acc_rate'].append(acc_rate.cpu().numpy())
                else:
                    U = U_new

                U = project_to_unitarity(U)

                if return_langevin_hist and i%10==0 and j%10==0:
                    observable_hist['tau'].append((j+1)*dtau)
                    a, b = calc_observables(U, n_resample=n_resample)
                    observable_hist['avg_plaquette'].append(torch_to_np(a))
                    observable_hist['topo_susc'].append(torch_to_np(b))
                    
            if return_hist:
                observable_hist['t'].append(torch_to_np(self.t_steps[-i-1]))
                a, b, c = calc_observables(U, give_q_avg=True, n_resample=n_resample)
                observable_hist['avg_plaquette'].append(torch_to_np(a))
                observable_hist['topo_susc'].append(torch_to_np(b))
                observable_hist['q_avg'].append(torch_to_np(c))
                loops = calc_wilson_loop_results(U, [[2,2], [3,3], [4,4]], n_resample=n_resample)
                observable_hist['2x2'].append(torch_to_np(loops[0]))
                observable_hist['3x3'].append(torch_to_np(loops[1]))
                observable_hist['4x4'].append(torch_to_np(loops[2]))

            if return_langevin_hist and i%10==0:
                observable_hist['avg_plaquette'] = np.array(observable_hist['avg_plaquette'])
                observable_hist['topo_susc'] = np.array(observable_hist['topo_susc'])
                hists.append(observable_hist)
                t_of_hists.append(torch_to_np(self.t_steps[-i-1]))

            if return_topo_charge_hist:
                t_hist.append(torch_to_np(self.t_steps[-i-1]))
                topo_charge_hist = torch.cat((topo_charge_hist, topological_charge(U)[None,:]), dim=0)
                print(i)

            if save_intermediate_configs:
                t_hist.append(torch_to_np(self.t_steps[-i-1]))
                torch.save(U, config_path+f'configs_{i}.pt')

        if save_intermediate_configs:
            torch.save(t_hist, config_path+'t_hist.pt')
            return U, acc_rate_hist
 
        if return_hist:
            observable_hist['avg_plaquette'] = np.array(observable_hist['avg_plaquette'])
            observable_hist['topo_susc'] = np.array(observable_hist['topo_susc'])
            observable_hist['q_avg'] = np.array(observable_hist['q_avg'])
            observable_hist['2x2'] = np.array(observable_hist['2x2'])
            observable_hist['3x3'] = np.array(observable_hist['3x3'])
            observable_hist['4x4'] = np.array(observable_hist['4x4'])
            return U, observable_hist, acc_rate_hist
        elif return_langevin_hist:
            return U, hists, t_of_hists, acc_rate_hist
        elif return_topo_charge_hist:
            return t_hist, torch_to_np(topo_charge_hist), acc_rate_hist
        else:
            return U, acc_rate_hist
    
    def solve_reverse_sde(self, 
                          n: int, 
                          n_steps: int, 
                          log_spaced_t=True, 
                          return_hist=False, 
                          n_resample=100, 
                          final_sq_steps=False, 
                          n_sq_steps=2, 
                          beta=2, 
                          dtau=0.005, 
                          progress_bar=False):
        if return_hist:
            observable_hist = {'t': [], 'avg_plaquette': [], 'topo_susc': [], 'q_avg': [], '2x2': [], '3x3': [], "4x4": []}
        
        U = self.generate_init_config(n)
        if log_spaced_t:
            reverse_t_steps = torch.exp(torch.linspace(np.log(1.),np.log(1e-3),n_steps))
        else:
            reverse_t_steps = torch.linspace(1.,0.,n_steps)

        if progress_bar:
            epochs = tqdm.tqdm(range(n_steps-1))
        else: 
            epochs = range(n_steps-1)

        for i in epochs:
            current_t = reverse_t_steps[i]
            current_dt = reverse_t_steps[i] - reverse_t_steps[i+1]

            if return_hist:
                observable_hist['t'].append(torch_to_np(current_t))
                a, b, c = calc_observables(U, give_q_avg=True, n_resample=n_resample)
                observable_hist['avg_plaquette'].append(torch_to_np(a))
                observable_hist['topo_susc'].append(torch_to_np(b))
                observable_hist['q_avg'].append(torch_to_np(c))
                loops = calc_wilson_loop_results(U, [[2,2], [3,3], [4,4]], n_resample=n_resample)
                observable_hist['2x2'].append(torch_to_np(loops[0]))
                observable_hist['3x3'].append(torch_to_np(loops[1]))
                observable_hist['4x4'].append(torch_to_np(loops[2]))
            
            t_batch = torch.ones(n) * current_t
            with torch.no_grad():
                score = self.network(U, t_batch).to(torch.cfloat)

            sigma = self.sde_param**current_t
            drift_term = - (-current_dt) * sigma**2 * score
            diffusion_factor = torch.sqrt(current_dt) * sigma * torch.ones(n)
            U = self.diffusion_step(U, diffusion_factor, drift_term=drift_term)
            U = project_to_unitarity(U)

        # Run some SQ with Metropolis as final steps to correct possible biases
        if final_sq_steps:
            for i in range(n_sq_steps):
                score = - wilson_action_grad(U, beta)
                diffusion_factor = np.sqrt(2.*dtau) * torch.ones(n)
                U_new = self.diffusion_step(U, diffusion_factor, drift_term=dtau*score)
                score_new = - wilson_action_grad(U_new, beta)
                U = self.metropolis_step(beta, U, U_new, score, score_new, dtau, get_acc_rate=False)
                U = project_to_unitarity(U)

        if return_hist:
            observable_hist['t'].append(torch_to_np(reverse_t_steps[-1]))
            a, b, c = calc_observables(U, give_q_avg=True, n_resample=n_resample)
            observable_hist['avg_plaquette'].append(torch_to_np(a))
            observable_hist['topo_susc'].append(torch_to_np(b))
            observable_hist['q_avg'].append(torch_to_np(c))
            loops = calc_wilson_loop_results(U, [[2,2], [3,3], [4,4]], n_resample=n_resample)
            observable_hist['2x2'].append(torch_to_np(loops[0]))
            observable_hist['3x3'].append(torch_to_np(loops[1]))
            observable_hist['4x4'].append(torch_to_np(loops[2]))

            observable_hist['avg_plaquette'] = np.array(observable_hist['avg_plaquette'])
            observable_hist['topo_susc'] = np.array(observable_hist['topo_susc'])
            observable_hist['q_avg'] = np.array(observable_hist['q_avg'])
            observable_hist['2x2'] = np.array(observable_hist['2x2'])
            observable_hist['3x3'] = np.array(observable_hist['3x3'])
            observable_hist['4x4'] = np.array(observable_hist['4x4'])
            return U, observable_hist
        else:
            return U
    
    def solve_reverse_ode(self, 
                          n: int, 
                          n_steps: int,
                          ode_time=1., 
                          return_hist=False, 
                          n_resample=100, 
                          progress_bar=False):
        if return_hist:
            observable_hist = {'t': [], 'avg_plaquette': [], 'topo_susc': [], 'q_avg': [], '2x2': [], '3x3': [], "4x4": []}
        
        U = self.generate_init_config(n)
        reverse_t_steps = torch.exp(torch.linspace(np.log(1.),np.log(1e-3),n_steps))
        #reverse_t_steps = torch.linspace(1.,1e-3,n_steps)


        if progress_bar:
            epochs = tqdm.tqdm(range(n_steps-1))
        else: 
            epochs = range(n_steps-1)

        for i in epochs:
            current_t = reverse_t_steps[i]
            current_dt = reverse_t_steps[i] - reverse_t_steps[i+1]

            if return_hist:
                observable_hist['t'].append(torch_to_np(current_t))
                a, b, c = calc_observables(U, give_q_avg=True, n_resample=n_resample)
                observable_hist['avg_plaquette'].append(torch_to_np(a))
                observable_hist['topo_susc'].append(torch_to_np(b))
                observable_hist['q_avg'].append(torch_to_np(c))
                loops = calc_wilson_loop_results(U, [[2,2], [3,3], [4,4]], n_resample=n_resample)
                observable_hist['2x2'].append(torch_to_np(loops[0]))
                observable_hist['3x3'].append(torch_to_np(loops[1]))
                observable_hist['4x4'].append(torch_to_np(loops[2]))
            
            t_batch = torch.ones(n) * current_t
            with torch.no_grad():
                score = self.network(U, t_batch).to(torch.cfloat)

            
            sigma = self.sde_param**current_t
            if current_t < ode_time: # Solve ODE (for a specified time interval)
                drift_term = current_dt * sigma**2  * score / 2.
                diffusion_factor = None
            else: # Solve SDE
                drift_term = current_dt * sigma**2  * score
                diffusion_factor = torch.sqrt(current_dt) * sigma * torch.ones(n)
            U = self.diffusion_step(U, diffusion_factor, drift_term=drift_term)
            
            U = project_to_unitarity(U)
            
        if return_hist:
            observable_hist['t'].append(torch_to_np(reverse_t_steps[-1]))
            a, b, c = calc_observables(U, give_q_avg=True, n_resample=n_resample)
            observable_hist['avg_plaquette'].append(torch_to_np(a))
            observable_hist['topo_susc'].append(torch_to_np(b))
            observable_hist['q_avg'].append(torch_to_np(c))
            loops = calc_wilson_loop_results(U, [[2,2], [3,3], [4,4]], n_resample=n_resample)
            observable_hist['2x2'].append(torch_to_np(loops[0]))
            observable_hist['3x3'].append(torch_to_np(loops[1]))
            observable_hist['4x4'].append(torch_to_np(loops[2]))

            observable_hist['avg_plaquette'] = np.array(observable_hist['avg_plaquette'])
            observable_hist['topo_susc'] = np.array(observable_hist['topo_susc'])
            observable_hist['q_avg'] = np.array(observable_hist['q_avg'])
            observable_hist['2x2'] = np.array(observable_hist['2x2'])
            observable_hist['3x3'] = np.array(observable_hist['3x3'])
            observable_hist['4x4'] = np.array(observable_hist['4x4'])
            return U, observable_hist
        else:
            return U

    def run_std_sq(self, 
                   U: torch.tensor, 
                   beta: float, 
                   n_steps: int, 
                   dtau: float, 
                   return_hist=False, 
                   n_resample=100, 
                   progress_bar=False):
        # Stochastic Quantiation with Metropolis adjustment, 
        # using the Wilson action at beta, using U as initial condition

        n = U.shape[0]

        if return_hist:
            observable_hist = {'i': [], 'avg_plaquette': [], 'topo_susc': [], 'q_avg': [], '2x2': [], '3x3': [], "4x4": []}

        if progress_bar:
            epochs = tqdm.tqdm(range(n_steps))
        else: 
            epochs = range(n_steps)

        for i in epochs:
            score = - wilson_action_grad(U, beta)
            diffusion_factor = np.sqrt(2.*dtau) * torch.ones(n)
            U_new = self.diffusion_step(U, diffusion_factor, drift_term=dtau*score)
            score_new = - wilson_action_grad(U_new, beta)
            U = self.metropolis_step(beta, U, U_new, score, score_new, dtau, get_acc_rate=False)
            U = project_to_unitarity(U)

            if return_hist:
                observable_hist['i'].append(i+1)
                a, b, c = calc_observables(U, give_q_avg=True, n_resample=n_resample)
                observable_hist['avg_plaquette'].append(torch_to_np(a))
                observable_hist['topo_susc'].append(torch_to_np(b))
                observable_hist['q_avg'].append(torch_to_np(c))
                loops = calc_wilson_loop_results(U, [[2,2], [3,3], [4,4]], n_resample=n_resample)
                observable_hist['2x2'].append(torch_to_np(loops[0]))
                observable_hist['3x3'].append(torch_to_np(loops[1]))
                observable_hist['4x4'].append(torch_to_np(loops[2]))

        if return_hist:
            observable_hist['avg_plaquette'] = np.array(observable_hist['avg_plaquette'])
            observable_hist['topo_susc'] = np.array(observable_hist['topo_susc'])
            observable_hist['q_avg'] = np.array(observable_hist['q_avg'])
            observable_hist['2x2'] = np.array(observable_hist['2x2'])
            observable_hist['3x3'] = np.array(observable_hist['3x3'])
            observable_hist['4x4'] = np.array(observable_hist['4x4'])
            return U, observable_hist
        else:
            return U

    def get_forward_diffusion_hist(self, U_0: torch.Tensor):
        observable_hist = {'t': [], 'avg_plaquette': [], 'topo_susc': [], 'q_avg': []}
        
        for i in range(self.t_n):
            p0t_std_batch = torch.ones(U_0.shape[0]) * self.p0t_std[i]
            print(i)

            if self.sde_type=='vp':
                p0t_mean_factor_batch = torch.ones(U_0.shape[0]) * self.p0t_mean_factor[i]
                U_0_scaled = fast_matrix_exp(fast_matrix_log(U_0) * p0t_mean_factor_batch[:,None,None,None,None,None])
                U_t = self.diffusion_step(U_0_scaled, p0t_std_batch)
            else:
                U_t = self.diffusion_step(U_0, p0t_std_batch)
            
            observable_hist['t'].append(torch_to_np(self.t_steps[i]))
            a, b, c = calc_observables(U_t, give_q_avg=True)
            observable_hist['avg_plaquette'].append(torch_to_np(a))
            observable_hist['topo_susc'].append(torch_to_np(b))
            observable_hist['q_avg'].append(torch_to_np(c))
        
        observable_hist['avg_plaquette'] = np.array(observable_hist['avg_plaquette'])
        observable_hist['topo_susc'] = np.array(observable_hist['topo_susc'])
        observable_hist['q_avg'] = np.array(observable_hist['q_avg'])
        return observable_hist
