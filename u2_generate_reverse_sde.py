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

if config["sde"]["lattice_size"][0] != 16:
    network.update_dims(config["sde"]["lattice_size"])
# Create diffusion model class
my_dm = u2_dm(network, 
            grid_size=config["sde"]["lattice_size"][0], 
            sde_type=network_config["dm"]["sde_type"], 
            sde_param=network_config["dm"]["sde_param"], 
            t_max=network_config["dm"]["t_max"], 
            t_n=network_config["dm"]["t_n"],
            use_log_spaced_t=network_config["dm"]["use_log_spaced_t"],
            t_log_min=network_config["dm"]["t_log_min"],
            t_log_max=network_config["dm"]["t_log_max"])

# Generate configurations
print("Generating configurations (SDE)...")
start = time.time()
result = my_dm.solve_reverse_sde(n=config["sde"]["n"],
                                 n_steps=config["sde"]["n_steps"], 
                                 return_hist=config["sde"]["return_hist"], 
                                 n_resample=config["sde"]["n_resample"], 
                                 progress_bar=config["sde"]["progress_bar"])
end = time.time()
print(f"Generation completed in {timedelta(seconds=(end-start))}")

if config["sde"]["return_hist"]:
    generated_samples, hist = result
    torch.save(hist, os.path.join(outdir,"sde_hist.pt"))
else:
    generated_samples = result

torch.save(generated_samples, os.path.join(outdir,"samples.pt"))