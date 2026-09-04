import argparse
from models import SinePINN, PolynomPINN, LiouvillePINN, LDFPINN, BSFPINN, BSWPINN
from models import SinePIGNN, PolynomPIGNN, LiouvillePIGNN, LDFPIGNN, BSFPIGNN, BSWPIGNN
from datasets import SineDataset, PolynomDataset, LiouvilleDataset
from datasets import LDFDataset, BSFDataset, BSWDataset
from dns_solvers import SinePoissonFDMSolver, PolynomPoissonFDMSolver, LiouvilleFDMSolver
from dns_solvers import LDFFDMSolver, BSFFDMSolver, BSWNSFVMSolver

data_root = 'data'
config_root = 'configs'
simulators = {'sine': SinePoissonFDMSolver,
              'polynom': PolynomPoissonFDMSolver,
              'liouville': LiouvilleFDMSolver,
              'lid_driven': LDFFDMSolver,
              'bward_step': BSFFDMSolver,
              'bowshock': BSWNSFVMSolver}
datasets = {'sine': SineDataset,
            'polynom': PolynomDataset,
            'liouville': LiouvilleDataset,
            'lid_driven': LDFDataset,
            'bward_step': BSFDataset,
            'bowshock': BSWDataset}
models = {'sine': {'pinn_c': SinePINN,
                   'pinn_d': SinePINN,
                   'pignn': SinePIGNN},
          'polynom': {'pinn_c': PolynomPINN,
                      'pinn_d': PolynomPINN,
                      'pignn': PolynomPIGNN},
          'liouville': {'pinn_c': LiouvillePINN,
                        'pinn_d': LiouvillePINN,
                        'pignn': LiouvillePIGNN},
          'lid_driven': {'pinn_c': LDFPINN,
                         'pinn_d': LDFPINN,
                         'pignn': LDFPIGNN},
          'bward_step': {'pinn_c': BSFPINN,
                         'pinn_d': BSFPINN,
                         'pignn': BSFPIGNN},
          'bowshock': {'pinn_c': BSWPINN,
                       'pinn_d': BSWPINN,
                       'pignn': BSWPIGNN}}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='pinn_c', choices=['pignn', 'pinn_d', 'pignn'],
                    help='Model type: pinn_c (PINN-continuous, auto-diff), pinn_d (PINN-discrete, num-diff), pignn (Physics-Informed GNN).')
parser.add_argument('-pde', '--pde', type=str, default='sine',
                    choices=['sine', 'polynom', 'liouville', 'lid_driven', 'bward_step', 'bowshock'])
parser.add_argument('-me', '--max_epoch', type=int, default=20000, help='maximum number of epoch')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('-ss', '--step_size', type=int, default=1000, help='parameter of step_size in scheduler optimizer.')
parser.add_argument('-g', '--gamma', type=float, default=0.9, help='parameter of gamma in scheduler optimizer.')
parser.add_argument('-bs', '--batch_size', type=int, default=-1,
                    help='only support batch training sine, polynom, and liouville for pinn_c (-1 for full batch).')
parser.add_argument('-a', '--alpha', type=float, default=1, help='weight of boundary loss.')
parser.add_argument('-b', '--beta', type=float, default=1, help='weight of pde loss.')
parser.add_argument('--early_stop', action='store_true', default=False,
                    help='apply early stop with patience of tolerant epoches.')
parser.add_argument('-threshold', '--safe_threshold', type=float, default=1e-5,
                    help='only start early stop when loss below this threshold.')
parser.add_argument('-pat', '--patience', type=int, default=100, help='patient epoches for early stop trigger.')
parser.add_argument('-tol', '--tolerance', type=float, default=1e-6, help='tolerance for early stop trigger.')
