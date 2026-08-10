import os
import numpy as np
import torch
import yaml
from datetime import datetime, timedelta
import time
import argparse

from u2_diffusion_model import u2_dm
from u2_networks import LCNN

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="GPU number", default=0)
parser.add_argument("--config", help="Path to config file", default="u2_dm_config.yaml")
args = parser.parse_args()
gpu_index = args.gpu
config_path = args.config

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

device = 'cuda:'+str(gpu_index)
torch.set_default_device(device)

# Create output folder
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
outdir = os.path.join(config["paths"]["result_path"], "generativerun-"+run_name)
os.makedirs(outdir)

print('+++++++++++++++++++++++++++++++++')
print(f"---- This is run {outdir} ----")

with open(os.path.join(outdir,"generate_config.yaml"),"w") as file:
    yaml.safe_dump(config, file)

# Load trained network and its config file
network = torch.load(config["paths"]["network_path"], map_location=device, weights_only=False)
with open(config["paths"]["network_config_path"], "r") as file:
    network_config = yaml.safe_load(file)
print("Load trained network")

with open(os.path.join(outdir,"network_config.yaml"),"w") as file:
    yaml.safe_dump(network_config, file)

if config["maala"]["lattice_size"][0] != 16:
    network.update_dims(config["maala"]["lattice_size"])
# Create diffusion model class
my_dm = u2_dm(network, 
            grid_size=config["maala"]["lattice_size"][0], 
            sde_type=network_config["dm"]["sde_type"], 
            sde_param=network_config["dm"]["sde_param"], 
            t_max=network_config["dm"]["t_max"], 
            t_n=network_config["dm"]["t_n"],
            use_log_spaced_t=config["maala"]["use_log_spaced_t"],
            t_log_min=network_config["dm"]["t_log_min"],
            t_log_max=network_config["dm"]["t_log_max"])

# Generate configurations
print("Generating configurations...")
start = time.time()
result = my_dm.maala_sampler(n=config["maala"]["n"], 
                            beta=config["maala"]["beta"], 
                            beta_0=config["maala"]["beta_0"], 
                            dtau=config["maala"]["dtau"], 
                            n_tau=config["maala"]["n_tau"], 
                            metropolis_threshold=config["maala"]["metropolis_threshold"], 
                            analytic_action_step=config["maala"]["analytic_action_step"], 
                            return_hist=config["maala"]["return_hist"],
                            return_topo_charge_hist=config["maala"]["return_topo_charge_hist"],
                            save_intermediate_configs=config["maala"]["save_intermediate_configs"], 
                            folder_path=outdir,
                            progress_bar=config["maala"]["progress_bar"],
                            n_resample=config["maala"]["n_resample"])
end = time.time()
print(f"Generation completed in {timedelta(seconds=(end-start))}")

if config["maala"]["return_hist"]:
    generated_samples, hist, acc_rate_hist = result
    torch.save(acc_rate_hist, os.path.join(outdir,"acc_rate_hist.pt"))
    torch.save(hist, os.path.join(outdir,"maala_hist.pt"))
elif config["maala"]["return_topo_charge_hist"]:
    generated_samples, topo_charge_hist, acc_rate_hist = result
    torch.save(acc_rate_hist, os.path.join(outdir,"acc_rate_hist.pt"))
    torch.save(topo_charge_hist, os.path.join(outdir,"topo_charge_hist.pt"))
else:
    generated_samples, acc_rate_hist = result
    torch.save(acc_rate_hist, os.path.join(outdir,"acc_rate_hist.pt"))

torch.save(generated_samples, os.path.join(outdir,"samples.pt"))