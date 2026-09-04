# A Systematic Analysis of Automatic Differentiation versus Discretization-based Constraints for Physics-Informed PDE Solvers
## PDE cases
**Poisson**
* Linear (Sine): -∇²u = 2π²·sin(πx)·sin(πy)
* Nonlinear (Polynomial): -∇²u = -4·(u + u³)
* Stiff (Liouville): -∇²u = 2·exp(u)

**Incompressible NS & Compressible Euler**
* Lid driven cavity flow
* Backward-facing step flow
* Hypersonic inviscid cylinder flow (bowshock)

## Requirements
* python 3.8.20
* torch 2.0.0+cu118
* torch-geometric 2.6.1
* numpy 1.24.4
* scipy 1.10.1
* scikit-learn 1.3.2
* cupy-cuda11x 12.3.0
* triangle 20250106
* networkx 2.8.2
* yaml 0.2.5

## Usage
We recommend starting from `train.py`, which is the entry point of the codebase. 
It loads the configuration file from `configs/`, reads training hyperparameters from `params.py`, 
and specifies the simulator to load datasets from `dataset.py` that 
provide discretization-based constraint information required by PINN<sub>d</sub> and PIGNN training, 
and the same discrete sampling points for PINN<sub>c</sub>.

We provide our source code of numerical solvers in package `dns_solvers/` and the neural models in package `models/`, 
with all problem definitions (e.g., mesh generation, boundary conditions, etc.) implemented in `dns_solvers/pdes.py`, 
and all simulation (FDM & FVM) implemented in `dns_solvers/simulator.py`.


## Examples
Here are some examples for running different test cases:
```
# Poisson cases:
python train.py -m pinn_c/pinn_d/pignn -pde sine/polynomial/liouville -a 0. -b 1.

# Lid driven cavity flow:
python train.py -m pinn_c/pinn_d/pignn -pde lid_driven -a 0. -b 1.

# Backward-facing step flow:
python train.py -m pinn_c/pinn_d/pignn -pde bward_step -a 1. -b 1.

# Hypersonic inviscid cylinder flow:
python train.py -m pinn_c/pinn_d/pignn -pde bowshock -a 1. -b 1.
```
Adjust model and PDE settings in configuration files located in `configs/`.
