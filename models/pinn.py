import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class BasePINN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc):
        super().__init__()
        actions = {'relu': nn.ReLU(),
                   'leaky_relu': nn.LeakyReLU(0.05),
                   'gelu': nn.GELU(),
                   'tanh': nn.Tanh(),
                   'swish': nn.SiLU()}
        self.num_layers = num_layers
        self.bias = bias
        self.act = actions[act] if act in actions.keys() else act
        self.discrete = discrete
        self.hard_bc = hard_bc
        self.layers = self._build_layers(input_dim, output_dim, hidden_units)
        self._initialize_weights()

    def _build_layers(self, input_dim, output_dim, hidden_units):
        layers = nn.ModuleList()
        if self.num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim, bias=self.bias))
        elif self.num_layers > 1:
            layers.append(nn.Linear(input_dim, hidden_units, bias=self.bias))
            for l in range(1, self.num_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units, bias=self.bias))
            layers.append(nn.Linear(hidden_units, output_dim, bias=self.bias))
        else:
            raise ValueError(f'Invalid number of network layers! Must be >= 1, got {self.num_layers}!')
        return layers

    def _initialize_weights(self):
        # initialize weight parameters using Xavier initialization
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _hard_constraint(self, U, *args, **kwargs):
        raise NotImplementedError('An implementation of _hard_constraint method is required!')

    def _partial_term(self, U, *args, **kwargs):
        """
        Calculate partial derivatives via automatic or numerical differentiation.
        https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
        https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/higher-order-gradients
        """
        raise NotImplementedError('An implementation of _partial_term method is required!')

    def residual(self, U, partial_term, *args, **kwargs):
        raise NotImplementedError('An implementation of residual method is required!')

    def forward(self, *args, **kwargs):
        """
        :param args: List of tensors (e.g., x, y, z, t. shape=[batch, 1])
        :param kwargs: required keyword arguments (e.g., coef_matrix, bound_label, etc.)
        :return: -Training mode: predictions and partial_derivatives
                 -Evaluation mode: predictions only
        """
        if self.training and not self.discrete:
            for arg in args:
                arg.requires_grad_(True)
        X = torch.cat(args, dim=-1)
        if self.num_layers == 1:
            U = self.layers[0](X)
        else:
            z = self.layers[0](X)
            for l in range(1, self.num_layers - 1):
                z = self.act(z)
                z = self.layers[l](z)
            z = self.act(z)
            U = self.layers[-1](z)
            if self.hard_bc:
                U = self._hard_constraint(U, *args, **kwargs)
        if self.training:
            partial_term = self._partial_term(U, *args, **kwargs)
            return U, partial_term
        else:
            return U


class SinePINN(BasePINN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete=False, hard_bc=True):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc)

    def _hard_constraint(self, u, *args, **kwargs):
        x, y = args
        hard_multiplier = x * y * (1 - x) * (1 - y)
        return u * hard_multiplier

    def _numerical_diff(self, coef_matrix, u):
        return torch.sparse.mm(coef_matrix, u)

    def _auto_diff(self, x, y, u):
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        return -(u_xx + u_yy)

    def _partial_term(self, u, *args, **kwargs):
        if self.discrete:
            coef_matrix = kwargs.get('coef_matrix')
            if coef_matrix is None:
                raise ValueError(f'Parameter of coef_matrix is required for discrete mode!')
            laplace = self._numerical_diff(coef_matrix, u)
        else:
            x, y = args
            laplace = self._auto_diff(x, y, u)
        return laplace

    def residual(self, u, partial_term, *args, **kwargs):
        x, y = args
        bound_label = kwargs.get('bound_label')
        bound_res = u[bound_label] - 0.0
        pde_res = partial_term[~bound_label] - 2 * (torch.pi ** 2) * torch.sin(
            torch.pi * x[~bound_label]) * torch.sin(torch.pi * y[~bound_label])
        return (torch.mean(bound_res ** 2), torch.mean(pde_res ** 2))


class PolynomPINN(BasePINN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete=False, hard_bc=True):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc)

    def _hard_constraint(self, u, *args, **kwargs):
        x, y = args
        a = torch.pi / 6.0
        d = (x + a) * (x - a) * (y + a) * (y - a)
        g = torch.tan(x + y)
        return u * d + g * (1.0 - d)

    def _numerical_diff(self, coef_matrix, u):
        return torch.sparse.mm(coef_matrix, u)

    def _auto_diff(self, x, y, u):
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        return -(u_xx + u_yy)

    def _partial_term(self, u, *args, **kwargs):
        if self.discrete:
            coef_matrix = kwargs.get('coef_matrix')
            if coef_matrix is None:
                raise ValueError(f'Parameter of coef_matrix is required for discrete mode!')
            laplace = self._numerical_diff(coef_matrix, u)
        else:
            x, y = args
            laplace = self._auto_diff(x, y, u)
        return laplace

    def residual(self, u, partial_term, *args, **kwargs):
        x, y = args
        bound_label = kwargs.get('bound_label')
        bound_res = u[bound_label] - (torch.tan(x + y))[bound_label]
        pde_res = partial_term[~bound_label] + 4 * (u[~bound_label] + u[~bound_label] ** 3)
        return (torch.mean(bound_res ** 2), torch.mean(pde_res ** 2))


class LiouvillePINN(BasePINN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete=False, hard_bc=True):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc)

    def _hard_constraint(self, u, *args, **kwargs):
        x, y = args
        hard_multiplier = x ** 2 + y ** 2 - 1
        return u * hard_multiplier

    def _numerical_diff(self, coef_matrix, u):
        return torch.sparse.mm(coef_matrix, u)

    def _auto_diff(self, x, y, u):
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        return u_xx + u_yy

    def _partial_term(self, u, *args, **kwargs):
        if self.discrete:
            coef_matrix = kwargs.get('coef_matrix')
            if coef_matrix is None:
                raise ValueError(f'Parameter of coef_matrix is required for discrete mode!')
            laplace = self._numerical_diff(coef_matrix, u)
        else:
            x, y = args
            laplace = self._auto_diff(x, y, u)
        return laplace

    def residual(self, u, partial_term, *args, **kwargs):
        bound_label = kwargs.get('bound_label')
        bound_res = u[bound_label] - 0.0
        pde_res = partial_term[~bound_label] + 2 * torch.exp(u[~bound_label])
        return (torch.mean(bound_res ** 2), torch.mean(pde_res ** 2))


class LDFPINN(BasePINN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete=False, hard_bc=True,
                 upwind=False):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc)
        self.upwind = upwind

    def _hard_constraint(self, U, *args, **kwargs):
        x, y = args
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        duv = x * y * (1.0 - x) * (1.0 - y)
        dp = x ** 2 + y ** 2
        u = u * duv + 4.0 * x * (1.0 - x) * y * (1.0 - duv)
        v = v * duv
        p = p * dp
        return torch.cat([u, v, p], dim=-1)

    def _numerical_diff(self, coef_matrices, U):
        if not self.upwind:
            u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
            M_gx, M_gy, _, _, M_v = coef_matrices
            l1 = (u * torch.sparse.mm(M_gx, u) + v * torch.sparse.mm(M_gy, u) + torch.sparse.mm(M_v, u) +
                  torch.sparse.mm(M_gx, p))
            l2 = (u * torch.sparse.mm(M_gx, v) + v * torch.sparse.mm(M_gy, v) + torch.sparse.mm(M_v, v) +
                  torch.sparse.mm(M_gy, p))
            l3 = torch.sparse.mm(M_gx, u) + torch.sparse.mm(M_gy, v)
            return (l1, l2, l3)
        else:
            # implement upwind matrix switching for high Re cases
            # see SINSFDMSolver._upwind_matrices() in dns_solvers for reference
            raise NotImplementedError

    def _auto_diff(self, x, y, U, **kwargs):
        Re = kwargs.get('Re')
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        grad_u = grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)
        grad_v = grad(v, [x, y], grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)
        grad_p = grad(p, [x, y], grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)
        u_x, u_y = grad_u[0], grad_u[1]
        v_x, v_y = grad_v[0], grad_v[1]
        p_x, p_y = grad_p[0], grad_p[1]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
        l1 = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
        l2 = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
        l3 = u_x + v_y
        return (l1, l2, l3)

    def _partial_term(self, U, *args, **kwargs):
        if self.discrete:
            coef_matrices = kwargs.get('coef_matrices')
            if coef_matrices is None:
                raise ValueError(f'Parameter of coef_matrices is required for discrete mode!')
            partial_term = self._numerical_diff(coef_matrices, U)
        else:
            x, y = args
            partial_term = self._auto_diff(x, y, U, **kwargs)
        return partial_term

    def residual(self, U, partial_term, *args, **kwargs):
        x, y = args
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        dirich_idx = kwargs.get('dirich_idx')
        bound_labs = kwargs.get('bound_labs')
        l1, l2, l3 = partial_term
        bound_res = (torch.mean((u - 4.0 * x * (1.0 - x) * y)[dirich_idx[0]] ** 2) + torch.mean(v[dirich_idx[0]] ** 2) +
                     torch.mean(p[dirich_idx[1]] ** 2))
        pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) +
                   torch.mean(l3[~bound_labs] ** 2))
        return (bound_res, pde_res)


class BSFPINN(BasePINN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete=False, hard_bc=True,
                 upwind=False):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc)
        self.upwind = upwind

    def _hard_constraint(self, U, *args, **kwargs):
        x, y = args
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        u = u * y * (2.0 - y)
        v = v * x * y * (2.0 - y)
        p = p * (4.0 - x)
        return torch.cat([u, v, p], dim=-1)

    def _numerical_diff(self, coef_matrices, U):
        if not self.upwind:
            u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
            M_gx, M_gy, _, _, M_v = coef_matrices
            l1 = (u * torch.sparse.mm(M_gx, u) + v * torch.sparse.mm(M_gy, u) + torch.sparse.mm(M_v, u) +
                  torch.sparse.mm(M_gx, p))
            l2 = (u * torch.sparse.mm(M_gx, v) + v * torch.sparse.mm(M_gy, v) + torch.sparse.mm(M_v, v) +
                  torch.sparse.mm(M_gy, p))
            l3 = torch.sparse.mm(M_gx, u) + torch.sparse.mm(M_gy, v)
            return (l1, l2, l3)
        else:
            # implement upwind matrix switching for high Re cases
            # see SINSFDMSolver._upwind_matrices() in dns_solvers for reference
            raise NotImplementedError

    def _auto_diff(self, x, y, U, **kwargs):
        Re = kwargs.get('Re')
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        grad_u = grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)
        grad_v = grad(v, [x, y], grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)
        grad_p = grad(p, [x, y], grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)
        u_x, u_y = grad_u[0], grad_u[1]
        v_x, v_y = grad_v[0], grad_v[1]
        p_x, p_y = grad_p[0], grad_p[1]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
        l1 = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
        l2 = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
        l3 = u_x + v_y
        return (l1, l2, l3)

    def _partial_term(self, U, *args, **kwargs):
        if self.discrete:
            coef_matrices = kwargs.get('coef_matrices')
            if coef_matrices is None:
                raise ValueError(f'Parameter of coef_matrices is required for discrete mode!')
            partial_term = self._numerical_diff(coef_matrices, U)
        else:
            x, y = args
            partial_term = self._auto_diff(x, y, U, **kwargs)
        return partial_term

    def residual(self, U, partial_term, *args, **kwargs):
        l1, l2, l3 = partial_term
        if self.discrete:
            x, y = args
            u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
            inlet_idx, wall_idx = kwargs.get('inlet_idx'), kwargs.get('wall_idx')
            outlet_idx, inner_outlet_idx = kwargs.get('outlet_idx'), kwargs.get('inner_outlet_idx')
            bound_labs = kwargs.get('bound_labs')
            # bound_res = (torch.mean((u - 4 * y * (1 - y))[inlet_idx] ** 2) + torch.mean(u[wall_idx] ** 2) +
            #              torch.mean(v[torch.cat([inlet_idx, wall_idx], dim=-1)] ** 2) +
            #              torch.mean(p[outlet_idx] ** 2) + torch.mean((u[outlet_idx] - u[inner_outlet_idx])**2) +
            #              torch.mean((v[outlet_idx] - v[inner_outlet_idx])**2))
            bound_res = (torch.mean((u - 4 * y * (1 - y))[inlet_idx] ** 2) + torch.mean(u[wall_idx] ** 2) +
                         torch.mean(v[torch.cat([inlet_idx, wall_idx], dim=-1)] ** 2) +
                         torch.mean(p[outlet_idx] ** 2))
            pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) +
                       torch.mean(l3[~bound_labs] ** 2))

        else:
            x, y = args
            u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
            inlet_idx, wall_idx, outlet_idx = kwargs.get('inlet_idx'), kwargs.get('wall_idx'), kwargs.get('outlet_idx')
            bound_labs = kwargs.get('bound_labs')
            bound_res = (torch.mean((u - 4 * y * (1 - y))[inlet_idx] ** 2) + torch.mean(u[wall_idx] ** 2) +
                         torch.mean(v[torch.cat([inlet_idx, wall_idx], dim=-1)] ** 2) +
                         torch.mean(p[outlet_idx] ** 2))
            pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) +
                       torch.mean(l3[~bound_labs] ** 2))
        return (bound_res, pde_res)


class BSWPINN(BasePINN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, discrete=False, hard_bc=True,
                 neumann=False, muscl=False):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, discrete, hard_bc)
        self.neumann = neumann
        self.muscl = muscl

    def _hard_constraint(self, U, *args, **kwargs):
        x, y = args
        rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
        rho = 1e-3 + F.softplus(rho)
        p = 1e-3 + F.softplus(p)
        m_inf = kwargs.get('m_inf')
        d_in = 3.4 ** 2 / 9.0 - (x - 1.6 / 3.0) ** 2 - y ** 2
        rho = rho * d_in + 1.0 * (1.0 - d_in)
        u = u * d_in + m_inf * (1.0 - d_in)
        v = v * d_in
        p = p * d_in + (1.0 - d_in) / 1.4
        return torch.cat([rho, u, v, p], dim=-1)

    def _scatter_aggregate(self, indices, values, minlength, aggr):
        if aggr not in ("add", "mean", "max", "min"):
            raise ValueError('Unrecognized aggregator type! Please select one from (add, mean, max, min)!')
        if values.dim() == 1:
            shape = (minlength,)
        else:
            shape = (minlength, values.shape[-1])
        if aggr == "add":
            output = torch.zeros(shape, device=values.device, dtype=values.dtype)
            output.scatter_add_(0, indices, values)
            return output
        if aggr == "mean":
            add = torch.zeros(shape, device=values.device, dtype=values.dtype)
            count = torch.zeros(shape, device=values.device, dtype=values.dtype)
            add.scatter_add_(0, indices, values)
            count.scatter_add_(0, indices, torch.ones_like(values))
            output = add / (count + 1e-10)
            return output
        if aggr == "max":
            output = torch.full(shape, -torch.inf, device=values.device, dtype=values.dtype)
            output.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
            return output
        if aggr == "min":
            output = torch.full(shape, torch.inf, device=values.device, dtype=values.dtype)
            output.scatter_reduce_(0, indices, values, reduce="amin", include_self=True)
            return output

    def _gradient_lsm(self, phi, lsm_neighbors, lsm_matrices):
        phi = phi.squeeze(-1)
        del_phi = phi[lsm_neighbors] - phi[:, None]
        grad_phi = torch.einsum('nij,nj->ni', lsm_matrices, del_phi)
        return grad_phi

    def _venkata_lim(self, phi, grad_phi, dual_centers, pos, edge_nodes):
        phi = phi.squeeze(-1)
        vi, vj = edge_nodes[:, 0], edge_nodes[:, 1]
        del_phi = phi[vj] - phi[vi]
        del_phi_i = torch.sum(grad_phi[vi] * (dual_centers - pos[vi]), dim=1)
        del_phi_j = torch.sum(grad_phi[vj] * (pos[vj] - dual_centers), dim=1)
        theta_i = del_phi / torch.where(del_phi_i >= 0, del_phi_i + 1e-10, del_phi_i - 1e-10)
        theta_j = del_phi / torch.where(del_phi_j >= 0, del_phi_j + 1e-10, del_phi_j - 1e-10)
        theta = torch.column_stack([theta_i, theta_j])
        theta_c = torch.maximum(
            (theta ** 2 + 2.0 * theta) / (theta ** 2 + theta + 2.0), torch.tensor(0.0, device=theta.device))
        weight = self._scatter_aggregate(edge_nodes.ravel(), theta_c.ravel(), phi.shape[0], 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _face_state(self, rho, u, v, p, *args, **kwargs):
        edge_nodes = kwargs.get('edge_nodes')
        vi, vj = edge_nodes[:, 0], edge_nodes[:, 1]
        if not self.muscl:
            E = p / (0.4 * rho) + 0.5 * (u ** 2 + v ** 2)
            T = 1.4 * p / rho
            c = torch.sqrt(T)
            rho_l, rho_r = rho[vi], rho[vj]
            u_l, u_r = u[vi], u[vj]
            v_l, v_r = v[vi], v[vj]
            p_l, p_r = p[vi], p[vj]
            c_l, c_r = c[vi], c[vj]
            E_l, E_r = E[vi], E[vj]
        else:
            dual_centers = kwargs.get('dual_centers')
            pos = torch.cat(args, dim=-1)
            lsm_neighbors, lsm_matrices = kwargs.get('lsm_neighbors'), kwargs.get('lsm_matrices')
            dfi, dfj = dual_centers - pos[vi], dual_centers - pos[vj]
            grad_rho = 3.0 * self._gradient_lsm(rho, lsm_neighbors, lsm_matrices)
            grad_u = 3.0 * self._gradient_lsm(u, lsm_neighbors, lsm_matrices)
            grad_v = 3.0 * self._gradient_lsm(v, lsm_neighbors, lsm_matrices)
            grad_p = 3.0 * self._gradient_lsm(p, lsm_neighbors, lsm_matrices)
            grad_rho = self._venkata_lim(rho, grad_rho, dual_centers, pos, edge_nodes)
            grad_u = self._venkata_lim(u, grad_u, dual_centers, pos, edge_nodes)
            grad_v = self._venkata_lim(v, grad_v, dual_centers, pos, edge_nodes)
            grad_p = self._venkata_lim(p, grad_p, dual_centers, pos, edge_nodes)
            rho_l = rho[vi] + torch.sum(grad_rho[vi] * dfi, dim=1, keepdim=True)
            u_l = u[vi] + torch.sum(grad_u[vi] * dfi, dim=1, keepdim=True)
            v_l = v[vi] + torch.sum(grad_v[vi] * dfi, dim=1, keepdim=True)
            p_l = p[vi] + torch.sum(grad_p[vi] * dfi, dim=1, keepdim=True)
            E_l = p_l / (0.4 * rho_l) + 0.5 * (u_l ** 2 + v_l ** 2)
            T_l = 1.4 * p_l / rho_l
            c_l = torch.sqrt(T_l)
            rho_r = rho[vj] + torch.sum(grad_rho[vj] * dfj, dim=1, keepdim=True)
            u_r = u[vj] + torch.sum(grad_u[vj] * dfj, dim=1, keepdim=True)
            v_r = v[vj] + torch.sum(grad_v[vj] * dfj, dim=1, keepdim=True)
            p_r = p[vj] + torch.sum(grad_p[vj] * dfj, dim=1, keepdim=True)
            E_r = p_r / (0.4 * rho_r) + 0.5 * (u_r ** 2 + v_r ** 2)
            T_r = 1.4 * p_r / rho_r
            c_r = torch.sqrt(T_r)
        vars_l = (rho_l, u_l, v_l, p_l, c_l, E_l)
        vars_r = (rho_r, u_r, v_r, p_r, c_r, E_r)
        return vars_l, vars_r

    def _riemann_flux(self, vars_l, vars_r, vi, vj, fnx, fny, dual_length, n_idx):
        rho_l, u_l, v_l, p_l, c_l, E_l = vars_l
        rho_r, u_r, v_r, p_r, c_r, E_r = vars_r
        un_l = u_l * fnx + v_l * fny
        un_r = u_r * fnx + v_r * fny
        Fa1_l = rho_l * un_l
        Fa2_l = rho_l * un_l * u_l + p_l * fnx
        Fa3_l = rho_l * un_l * v_l + p_l * fny
        Fa4_l = (rho_l * E_l + p_l) * un_l
        Fa1_r = rho_r * un_r
        Fa2_r = rho_r * un_r * u_r + p_r * fnx
        Fa3_r = rho_r * un_r * v_r + p_r * fny
        Fa4_r = (rho_r * E_r + p_r) * un_r
        lambda_max = torch.maximum(torch.abs(un_l) + c_l, torch.abs(un_r) + c_r)
        Fa1_f = dual_length * (0.5 * (Fa1_l + Fa1_r) - 0.5 * lambda_max * (rho_r - rho_l))
        Fa2_f = dual_length * (0.5 * (Fa2_l + Fa2_r) - 0.5 * lambda_max * (rho_r * u_r - rho_l * u_l))
        Fa3_f = dual_length * (0.5 * (Fa3_l + Fa3_r) - 0.5 * lambda_max * (rho_r * v_r - rho_l * v_l))
        Fa4_f = dual_length * (0.5 * (Fa4_l + Fa4_r) - 0.5 * lambda_max * (rho_r * E_r - rho_l * E_l))
        indices = torch.hstack([vi, vj])
        Fa1 = self._scatter_aggregate(indices, torch.hstack([Fa1_f.squeeze(-1), -Fa1_f.squeeze(-1)]), n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, torch.hstack([Fa2_f.squeeze(-1), -Fa2_f.squeeze(-1)]), n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, torch.hstack([Fa3_f.squeeze(-1), -Fa3_f.squeeze(-1)]), n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, torch.hstack([Fa4_f.squeeze(-1), -Fa4_f.squeeze(-1)]), n_idx, 'add')
        return torch.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _numerical_diff(self, U, *args, **kwargs):
        n_idx = U.shape[0]
        edge_nodes = kwargs.get('edge_nodes')
        dual_length = kwargs.get('dual_length')
        dual_norm = kwargs.get('dual_norm')
        dual_areas = kwargs.get('dual_areas')
        fnx, fny = dual_norm[:, 0:1], dual_norm[:, 1:2]
        vi, vj = edge_nodes[:, 0], edge_nodes[:, 1]
        rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
        vars_l, vars_r = self._face_state(rho, u, v, p, *args, **kwargs)
        Fa = self._riemann_flux(vars_l, vars_r, vi, vj, fnx, fny, dual_length, n_idx)
        F = Fa / dual_areas
        l1, l2, l3, l4 = F[:, 0:1], F[:, 1:2], F[:, 2:3], F[:, 3:4]
        return (l1, l2, l3, l4)

    def _auto_diff(self, x, y, U, **kwargs):
        rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
        grad_rho = grad(rho, [x, y], grad_outputs=torch.ones_like(rho), retain_graph=True, create_graph=True)
        grad_u = grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)
        grad_v = grad(v, [x, y], grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)
        grad_p = grad(p, [x, y], grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)
        # grad_T = grad(T, [x, y], grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)
        rho_x, rho_y = grad_rho[0], grad_rho[1]
        u_x, u_y = grad_u[0], grad_u[1]
        v_x, v_y = grad_v[0], grad_v[1]
        p_x, p_y = grad_p[0], grad_p[1]
        # T_x, T_y = grad_T[0], grad_T[1]
        # T_x, T_y = 1.4 * (rho * p_x - rho_x * p) / rho ** 2, 1.4 * (rho * p_y - rho_y * p) / rho ** 2
        l1 = rho * (u_x + v_y) + rho_x * u + rho_y * v
        l2 = (rho * (2 * u * u_x + u_y * v + u * v_y) + rho_x * u ** 2 + rho_y * u * v + p_x)
        l3 = (rho * (2 * v * v_y + u_x * v + u * v_x) + rho_x * u * v + rho_y * v ** 2 + p_y)
        l4 = (3.5 *(u_x * p + u * p_x) + 0.5 * rho_x * (u ** 3 + u * v ** 2) +
              0.5 * rho * (3.0 * u ** 2 * u_x + u_x * v ** 2 + 2.0 * u * v * v_x) +
              3.5 * (v_y * p + v * p_y) + 0.5 * rho_y * (u ** 2 * v + v ** 3) +
              0.5 * rho * (2.0 * u * v * u_y + u ** 2 * v_y + 3.0 * v ** 2 * v_y))
        return (l1, l2, l3, l4)

    def _partial_term(self, U, *args, **kwargs):
        if self.discrete:
            partial_term = self._numerical_diff(U, *args, **kwargs)
        else:
            x, y = args
            partial_term = self._auto_diff(x, y, U, **kwargs)
        return partial_term

    def residual(self, U, partial_term, *args, **kwargs):
        l1, l2, l3, l4 = partial_term
        if self.discrete:
            rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
            m_inf = kwargs.get('m_inf')
            surf_idx, inlet_idx, outlet_idx = kwargs.get('surf_idx'), kwargs.get('inlet_idx'), kwargs.get('outlet_idx')
            inner_surf_idx, inner_outlet_idx = kwargs.get('inner_surf_idx'), kwargs.get('inner_outlet_idx')
            surf_norm = kwargs.get('surf_norm')
            bound_labs = kwargs.get('bound_labs')
            if self.neumann:
                bound_res = (torch.mean((rho[inlet_idx] - 1) ** 2) + torch.mean((u[inlet_idx] - m_inf) ** 2) +
                             torch.mean((u[surf_idx] * surf_norm[:, 0: 1] + v[surf_idx] * surf_norm[:, 1:2]) ** 2) +
                             torch.mean(v[inlet_idx] ** 2) + torch.mean((p[inlet_idx] - 1.0 / 1.4) ** 2) +
                             torch.mean((rho[surf_idx] - rho[inner_surf_idx]) ** 2) +
                             torch.mean((p[surf_idx] - p[inner_surf_idx]) ** 2) +
                             torch.mean((rho[outlet_idx] - rho[inner_outlet_idx]) ** 2) +
                             torch.mean((u[outlet_idx] - u[inner_outlet_idx]) ** 2) +
                             torch.mean((v[outlet_idx] - v[inner_outlet_idx]) ** 2) +
                             torch.mean((p[outlet_idx] - p[inner_outlet_idx]) ** 2))
            else:
                bound_res = (torch.mean((rho[inlet_idx] - 1) ** 2) + torch.mean((u[inlet_idx] - m_inf) ** 2) +
                             torch.mean((u[surf_idx] * surf_norm[:, 0: 1] + v[surf_idx] * surf_norm[:, 1:2]) ** 2) +
                             torch.mean(v[inlet_idx] ** 2) + torch.mean((p[inlet_idx] - 1.0 / 1.4) ** 2))
            pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) / m_inf ** 2 +
                       torch.mean(l3[~bound_labs] ** 2) / m_inf ** 2 + torch.mean(l4[~bound_labs] ** 2) / m_inf ** 4)
        else:
            rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
            m_inf = kwargs.get('m_inf')
            surf_idx, inlet_idx = kwargs.get('surf_idx'), kwargs.get('inlet_idx')
            surf_norm = kwargs.get('surf_norm')
            bound_labs = kwargs.get('bound_labs')
            bound_res = (torch.mean((rho[inlet_idx] - 1.0) ** 2) + torch.mean((u[inlet_idx] - m_inf) ** 2) +
                         torch.mean(v[inlet_idx] ** 2) + torch.mean((p[inlet_idx] - 1.0 / 1.4) ** 2) +
                         torch.mean((u[surf_idx] * surf_norm[:, 0: 1] + v[surf_idx] * surf_norm[:, 1:2]) ** 2))
            pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) / m_inf ** 2 +
                       torch.mean(l3[~bound_labs] ** 2) / m_inf ** 2 + torch.mean(l4[~bound_labs] ** 2) / m_inf ** 4)
        return (bound_res, pde_res)
