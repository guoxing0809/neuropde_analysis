import os
import params
import utils
import torch
import time
import numpy as np
import torch.nn as nn
from datasets import DataLoader
from plot import *


def train():
    args = params.parser.parse_args()
    if (args.model != 'pinn_c' or args.pde not in ('sine', 'polynom', 'liouville')) and args.batch_size != -1:
        raise ValueError(
            f'Batch training is only available for poisson pdes with pinn_c, please set batch_size=-1 for full batch!')
    configs = utils.load_yaml_params(config_path=os.path.join(params.config_root, f'{args.pde}.yaml.'))
    simulator = params.simulators[args.pde]
    dataset = params.datasets[args.pde](simulator, **configs['data']['mesh'])
    train_data = dataset.load_dataset(model=args.model, data_file=configs['data']['data_file'])
    model = params.models[args.pde][args.model](**configs['model'][args.model])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    model.to(device)
    train_data = utils.tensor2device(train_data, device)
    patience = 0
    best_loss = float('inf')
    # loss_bound_items = []
    # loss_pde_items = []
    # loss_items = []
    # U_ref = dataset.simulator.reference_solution()
    # l2_items = []
    print(f'Training pde of {args.pde} on model {args.model} on device {device}......')
    start = time.perf_counter()
    for epoch in range(args.max_epoch):
        model.train()
        if args.model == 'pinn_c' and args.batch_size >= 1:
            loss = 0.
            loss_bound = 0.
            loss_pde = 0.
            for batch in DataLoader(train_data, batch_size=args.batch_size):
                U, partial_term = model(*batch['args'])
                bound_res, pde_res = model.residual(U, partial_term, *batch['args'], **batch['kwargs'])
                batch_res = args.alpha * bound_res + args.beta * pde_res
                optimizer.zero_grad()
                batch_res.backward()
                optimizer.step()
                scheduler.step()
                loss_bound += bound_res.item()
                loss_pde += pde_res.item()
                loss += batch_res.item()
            loss_bound /= len(train_data)
            loss_pde /= len(train_data)
            loss /= len(train_data)
        else:
            if args.model in ('pinn_c', 'pinn_d'):
                U, partial_term = model(*train_data['args'], **train_data['kwargs'])
                bound_res, pde_res = model.residual(U, partial_term, *train_data['args'], **train_data['kwargs'])
            else:
                U, partial_term = model(train_data['x'], train_data['edge_index'], **train_data['kwargs'])
                bound_res, pde_res = model.residual(U, partial_term, train_data['x'], **train_data['kwargs'])
            res = args.alpha * bound_res + args.beta * pde_res
            optimizer.zero_grad()
            res.backward()
            optimizer.step()
            scheduler.step()
            loss_bound = bound_res.item()
            loss_pde = pde_res.item()
            loss = res.item()
        # loss_bound_items.append(loss_bound)
        # loss_pde_items.append(loss_pde)
        # loss_items.append(loss)
        # l2 = utils.l2_norm_error(U.squeeze(-1).detach().cpu().numpy(), U_ref)
        # l2_items.append(l2)
        print(f'epoch {epoch}: loss_bound={loss_bound}, loss_pde={loss_pde}, loss={loss}.')
        if args.early_stop and loss <= args.safe_threshold:
            if loss < best_loss and best_loss - loss > args.tolerance:
                best_loss = loss
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print(f'Early stop triggered at epoch {epoch}!')
                    break
    end = time.perf_counter()
    cost = end - start
    print(f'cost time: {cost / 60:.2f}min!')
    model.eval()
    print(f'Evaluating pde of {args.pde} on model {args.model}......')
    with torch.no_grad():
        if args.model in ('pinn_c', 'pinn_d'):
            U_preds = model(*train_data['args'], **train_data['kwargs'])
        else:
            U_preds = model(train_data['x'], train_data['edge_index'], **train_data['kwargs'])
    U_preds = U_preds.squeeze(-1).detach().cpu().numpy()
    if configs['data']['ref_file'] is not None:
        ref_file = os.path.join(params.data_root, configs['data']['ref_file'])
        U_ref = dataset.simulator.reference_solution(ref_file)
    else:
        U_ref = dataset.simulator.reference_solution()
    pos = dataset.simulator.pos
    triangles = dataset.simulator.triangles if hasattr(dataset.simulator, 'triangles') else None
    if U_preds.ndim > 1:
        for i in range(U_preds.shape[1]):
            plot_irregular_contour2D(pos, U_preds[:, i], triangles=triangles)
    else:
        plot_irregular_contour2D(pos, U_preds, triangles=triangles)
    if args.pde != 'bowshock':
        l2_error = utils.l2_norm_error(U_preds, U_ref)
        inf_error = utils.inf_norm_error(U_preds, U_ref)
        print(f'l2_norm_error: {l2_error}, inf_norm_error: {inf_error}')
        # return l2_error, inf_error


if __name__ == '__main__':
    train()
    # l2_errors = []
    # inf_errors = []
    # for i in range(5):
    #     l2, inf = train()
    #     l2_errors.append(l2)
    #     inf_errors.append(inf)
    # l2_errors = np.vstack(l2_errors)
    # inf_errors = np.vstack(inf_errors)
    # print(np.column_stack([l2_errors, inf_errors]))
    # print(
    #     f'l2_norm_error: mean={np.mean(l2_errors, axis=0)}, std={np.std(l2_errors, axis=0)}; inf_norm_error: mean={np.mean(inf_errors, axis=0)}, std={np.std(inf_errors, axis=0)}')
