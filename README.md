# DM_U2_LGT
Code for the papers "Generalizable Equivariant Diffusion Models for Non-Abelian Lattice Gauge Theory", [arXiv:2601.19552](https://arxiv.org/abs/2601.19552) and "Generalizable Equivariant Diffusion Models for Non-Abelian Lattice Gauge Theory: Analytical Foundations and Numerical Scaling Tests", [arXiv:2611.03530](https://arxiv.org/abs/2611.03530).

Diffusion model for U(2) lattice gauge theory in 2D. Can be used to train an L-CNN using score matching, and then generate field configurations using either the reverse stochastic differential equation (SDE) or the Metropolis-adjusted annealed Langevin algorithm (MAALA).

# Install
```bash
conda env create -f env.yaml
```

# Run training
Create a config file following `example_training_configuration.yaml`. Then run 
```bash
python u2_training.py --config config_file.yaml
```

This will generate a folder containing the trained network and the config file.

# Generate field configurations
To generate using MAALA, follow `example_generate_maala_configuration.yaml` and then run
```bash
python u2_generate_maala.py --config config_file.yaml
```

To generate using the reverse SDE, follow `example_generate_reverse_sde_configuration.yaml` and then run
```bash
python u2_generate_reverse_sde.py --config config_file.yaml
```

# Example network
Examplarily, we also provide a trained network under `data/example_network`. This network was trained at $\beta=2$ and $L=16$. 
Using `generate_and_evaluate.ipynb`, one can generate configurations with this example network, and plot observables. 
