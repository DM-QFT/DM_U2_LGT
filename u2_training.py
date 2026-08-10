import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from datetime import datetime, timedelta
import time
import argparse

from u2_diffusion_model import u2_dm
from u2_networks import LCNN

# Load config
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
outdir = os.path.join(config["paths"]["result_path"], "trainingrun-"+run_name)
os.makedirs(outdir)
os.makedirs(outdir+'/ckpts/')

print('+++++++++++++++++++++++++++++++++')
print(f"---- This is run {outdir} ----")

with open(os.path.join(outdir,"training_config.yaml"),"w") as file:
    yaml.safe_dump(config, file)

training_samples = torch.load(config["training"]["training_data_path"], map_location=device).to(torch.cfloat)
data_set = TensorDataset(training_samples)
data_loader = DataLoader(data_set, batch_size=config["training"]["batch_size"], shuffle=True, generator=torch.Generator(device))

if config["paths"]["retrain"]:
    network = torch.load(config["paths"]["network_path"], map_location=device, weights_only=False)
    print("Load trained network")
else:
    network = LCNN(channels=config["network"]["channels"], 
                        n_layers=config["network"]["n_layers"],
                        kernel_sizes=config["network"]["kernel_sizes"],
                        log_time_embedding=config["network"]["log_time_embedding"],
                        time_embed_dim=config["network"]["time_embed_dim"],
                        time_embed_scale=config["network"]["time_embed_scale"],
                        shift_time_embedding=config["network"]["shift_time_embedding"]).to(device)

# Create diffusion model class
my_dm = u2_dm(network, 
            grid_size=16, 
            sde_type=config["dm"]["sde_type"], 
            sde_param=config["dm"]["sde_param"], 
            t_max=config["dm"]["t_max"], 
            t_n=config["dm"]["t_n"],
            use_log_spaced_t=config["dm"]["use_log_spaced_t"],
            t_log_min=config["dm"]["t_log_min"],
            t_log_max=config["dm"]["t_log_max"])

# Training
print("Training network...")
start = time.time()

if config["training"]["get_validation_loss"]:
    val_loss = my_dm.train(data_loader, 
                n_epochs=config["training"]["n_epochs"], 
                lr=config["training"]["lr"], 
                lr_decay=config["training"]["lr_decay"],
                lambda_term=config["training"]["lambda_term"],
                ckpt_interval=config["training"]["ckpt_interval"],
                ckpt_path=outdir+'/ckpts/',
                progress_bar=config["training"]["progress_bar"],
                get_validation_loss=True,
                validation_set=training_samples[:config["training"]["validation_batch_size"]].to(device))
    np.save(os.path.join(outdir,"loss_history"), np.array(val_loss))
else:
    my_dm.train(data_loader, 
                n_epochs=config["training"]["n_epochs"], 
                lr=config["training"]["lr"], 
                lr_decay=config["training"]["lr_decay"],
                lambda_term=config["training"]["lambda_term"],
                ckpt_interval=config["training"]["ckpt_interval"],
                ckpt_path=outdir+'/ckpts/',
                progress_bar=config["training"]["progress_bar"],
                get_validation_loss=False)

end = time.time()
print(f"Training completed in {timedelta(seconds=(end-start))}")
torch.save(my_dm.network, os.path.join(outdir,"network.pt"))


print('+++++++++++++++++++++++++++++++++')