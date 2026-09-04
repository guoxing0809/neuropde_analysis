import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import sort_edge_index, softmax


class BasePIGNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs):
        super().__init__()
        actions = {'relu': nn.ReLU(),
                   'leaky_relu': nn.LeakyReLU(0.05),
                   'gelu': nn.GELU(),
                   'tanh': nn.Tanh(),
                   'swish': nn.SiLU()}
        self.num_layers = num_layers
        self.bias = bias
        self.act = actions[act] if act in actions.keys() else act
        self.hard_bc = hard_bc
        self.gnn = gnn
        self.Conv = self._build_gnnconv(gnn, **kwargs)
        self.convs = self._build_layers(input_dim, output_dim, hidden_units, **kwargs)
        self._initialize_weights()

    def _build_gnnconv(self, gnn, **kwargs):
        allowed_gnns = {'gcn': GCNConv,
                        'gat': GATConv,
                        'sage': SAGEConv}
        if gnn not in allowed_gnns.keys():
            raise ValueError(f'Unrecognized gnn type {gnn}, please select one from (gcn, gat, sage)!')
        self.heads = kwargs.get('heads', None)
        self.aggr = kwargs.get('aggr', None)
        if gnn == 'gat' and self.heads is None:
            raise ValueError(f'Parameter of heads is required for GAT!')
        if gnn == 'sage' and self.aggr is None:
            raise ValueError(f'Parameter of aggr is required for SAGE!')
        return allowed_gnns[gnn]

    def _build_layers(self, input_dim, output_dim, hidden_units, **kwargs):
        convs = nn.ModuleList()
        if self.num_layers == 1:
            convs.append(self.Conv(input_dim, output_dim, bias=self.bias, **kwargs))
        elif self.num_layers > 1:
            convs.append(self.Conv(input_dim, hidden_units, bias=self.bias, **kwargs))
            for l in range(1, self.num_layers - 1):
                convs.append(self.Conv(hidden_units, hidden_units, bias=self.bias, **kwargs))
            convs.append(self.Conv(hidden_units, output_dim, bias=self.bias, **kwargs))
        else:
            raise ValueError(f'Invalid number of network layers! Must be >= 1, got {self.num_layers}!')
        return convs

    def _initialize_weights(self):
        # initialize weight parameters using Xavier initialization
        if self.gnn == 'gcn':
            for conv in self.convs:
                nn.init.xavier_uniform_(conv.lin.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        if self.gnn == 'gat':
            for conv in self.convs:
                nn.init.xavier_uniform_(conv.lin.weight)
                nn.init.xavier_uniform_(conv.att_src)
                nn.init.xavier_uniform_(conv.att_dst)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        if self.gnn == 'sage':
            for conv in self.convs:
                nn.init.xavier_uniform_(conv.lin_l.weight)
                nn.init.xavier_uniform_(conv.lin_r.weight)
                if conv.lin_l.bias is not None:
                    nn.init.zeros_(conv.lin_l.bias)
                if self.aggr == 'lstm':
                    for name, param in conv.aggr_module.lstm.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.zeros_(param)

    def _hard_constraint(self, x, u, **kwargs):
        raise NotImplementedError('An implementation of _hard_constraint method is required!')

    def _partial_term(self, x, u, **kwargs):
        """
        Calculate partial derivatives via numerical differentiation.
        """
        raise NotImplementedError('An implementation of _partial_term method is required!')

    def forward(self, x, edge_index, **kwargs):
        """
        :param x: input features, shape=[n, dim]
        :param edge_index: undirected graph (source, target), int64, shape=[2, num_edges]
        :param kwargs: keyword arguments (e.g., coef_matrix, sparse_dot) required for discrete mode
        :return: -Training mode: predictions and partial_derivatives
                 -Evaluation mode: predictions only
        """
        if self.gnn == 'sage' and self.aggr == 'lstm':
            edge_index = sort_edge_index(edge_index, sort_by_row=False)
            edge_index = edge_index.contiguous()
        if self.num_layers == 1:
            u = self.convs[0](x, edge_index)
        else:
            z = self.convs[0](x, edge_index)
            for l in range(1, self.num_layers - 1):
                z = self.act(z)
                z = self.convs[l](z, edge_index)
            z = self.act(z)
            u = self.convs[-1](z, edge_index)
            if self.hard_bc:
                u = self._hard_constraint(x, u, **kwargs)
        if self.training:
            partial_term = self._partial_term(x, u, **kwargs)
            return u, partial_term
        else:
            return u


class SinePIGNN(BasePIGNN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc=True, gnn='gcn', **kwargs):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs)

    def _hard_constraint(self, x, u, **kwargs):
        hard_multiplier = x[:, 0:1] * x[:, 1:2] * (1 - x[:, 0:1]) * (1 - x[:, 1:2])
        return u * hard_multiplier

    def _partial_term(self, x, u, **kwargs):
        coef_matrix = kwargs.get('coef_matrix')
        if coef_matrix is None:
            raise ValueError(f'Parameter of coef_matrix is required for discrete mode!')
        laplace = torch.sparse.mm(coef_matrix, u)
        return laplace

    def residual(self, u, partial_term, x, **kwargs):
        bound_label = kwargs.get('bound_label')
        bound_res = u[bound_label] - 0.0
        pde_res = partial_term[~bound_label] - 2 * (torch.pi ** 2) * torch.sin(
            torch.pi * x[~bound_label, 0:1]) * torch.sin(torch.pi * x[~bound_label, 1:2])
        return (torch.mean(bound_res ** 2), torch.mean(pde_res ** 2))


class PolynomPIGNN(BasePIGNN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc=True, gnn='gcn', **kwargs):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs)

    def _hard_constraint(self, x, u, **kwargs):
        a = torch.pi / 6.0
        d = (x[:, 0:1] + a) * (x[:, 0:1] - a) * (x[:, 1:2] + a) * (x[:, 1:2] - a)
        g = torch.tan(x[:, 0:1] + x[:, 1:2])
        return u * d + g * (1.0 - d)

    def _partial_term(self, x, u, **kwargs):
        coef_matrix = kwargs.get('coef_matrix')
        if coef_matrix is None:
            raise ValueError(f'Parameter of coef_matrix is required for discrete mode!')
        laplace = torch.sparse.mm(coef_matrix, u)
        return laplace

    def residual(self, u, partial_term, x, **kwargs):
        bound_label = kwargs.get('bound_label')
        bound_res = u[bound_label] - torch.tan(x[:, 0:1] + x[:, 1:2])[bound_label]
        pde_res = partial_term[~bound_label] + 4 * (u[~bound_label] + u[~bound_label] ** 3)
        return (torch.mean(bound_res ** 2), torch.mean(pde_res ** 2))


class LiouvillePIGNN(BasePIGNN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc='hard', gnn='gcn', **kwargs):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs)

    def _hard_constraint(self, x, u, **kwargs):
        hard_multiplier = x[:, 0:1] ** 2 + x[:, 1:2] ** 2 - 1
        return u * hard_multiplier

    def _partial_term(self, x, u, **kwargs):
        coef_matrix = kwargs.get('coef_matrix')
        if coef_matrix is None:
            raise ValueError(f'Parameter of coef_matrix is required for discrete mode!')
        laplace = torch.sparse.mm(coef_matrix, u)
        return laplace

    def residual(self, u, partial_term, x, **kwargs):
        bound_label = kwargs.get('bound_label')
        bound_res = u[bound_label]
        pde_res = partial_term[~bound_label] + 2 * torch.exp(u[~bound_label])
        return (torch.mean(bound_res ** 2), torch.mean(pde_res ** 2))


class LDFPIGNN(BasePIGNN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc='hard', upwind=False,
                 gnn='gcn', **kwargs):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs)
        self.upwind = upwind

    def _hard_constraint(self, x, U, **kwargs):
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        duv = x[:, 0:1] * x[:, 1:2] * (1 - x[:, 0:1]) * (1 - x[:, 1:2])
        dp = (x[:, 0:1] ** 2 + x[:, 1:2]**2)
        u = u * duv + 4 * x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1.0 - duv)
        v = v * duv
        p = p * dp
        return torch.cat([u, v, p], dim=-1)

    def _partial_term(self, x, U, **kwargs):
        coef_matrices = kwargs.get('coef_matrices')
        if coef_matrices is None:
            raise ValueError(f'Parameter of coef_matrices is required for discrete mode!')
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

    def residual(self, U, partial_term, x, **kwargs):
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        dirich_idx = kwargs.get('dirich_idx')
        bound_labs = kwargs.get('bound_labs')
        l1, l2, l3 = partial_term
        bound_res = (torch.mean((u - 4 * x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2])[dirich_idx[0]] ** 2) +
                     torch.mean(v[dirich_idx[0]] ** 2) + torch.mean(p[dirich_idx[1]] ** 2))
        pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) +
                   torch.mean(l3[~bound_labs] ** 2))
        return (bound_res, pde_res)


class BSFPIGNN(BasePIGNN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc='hard', upwind=False,
                 gnn='gcn', **kwargs):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs)
        self.upwind = upwind

    def _hard_constraint(self, x, U, **kwargs):
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        u = u * x[:, 1:2] * (2.0 - x[:, 1:2])
        v = v * x[:, 0:1] * x[:, 1:2] * (2.0 - x[:, 1:2])
        p = p * (4.0 - x[:, 0:1])
        return torch.cat([u, v, p], dim=-1)

    def _partial_term(self, x, U, **kwargs):
        coef_matrices = kwargs.get('coef_matrices')
        if coef_matrices is None:
            raise ValueError(f'Parameter of coef_matrices is required for discrete mode!')
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

    def residual(self, U, partial_term, x, **kwargs):
        u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
        l1, l2, l3 = partial_term
        inlet_idx, wall_idx = kwargs.get('inlet_idx'), kwargs.get('wall_idx')
        outlet_idx, inner_outlet_idx = kwargs.get('outlet_idx'), kwargs.get('inner_outlet_idx')
        bound_labs = kwargs.get('bound_labs')
        # bound_res = (torch.mean((u - 4 * x[:, 1:2] * (1 - x[:, 1:2]))[inlet_idx] ** 2) + torch.mean(u[wall_idx] ** 2) +
        #              torch.mean(v[torch.cat([inlet_idx, wall_idx], dim=-1)] ** 2) +
        #              torch.mean(p[outlet_idx] ** 2) + torch.mean((u[outlet_idx] - u[inner_outlet_idx]) ** 2) +
        #              torch.mean((v[outlet_idx] - v[inner_outlet_idx]) ** 2))
        bound_res = (torch.mean((u - 4 * x[:, 1:2] * (1 - x[:, 1:2]))[inlet_idx] ** 2) + torch.mean(u[wall_idx] ** 2) +
                     torch.mean(v[torch.cat([inlet_idx, wall_idx], dim=-1)] ** 2) +
                     torch.mean(p[outlet_idx] ** 2))
        pde_res = (torch.mean(l1[~bound_labs] ** 2) + torch.mean(l2[~bound_labs] ** 2) +
                   torch.mean(l3[~bound_labs] ** 2))
        return (bound_res, pde_res)



class BSWPIGNN(BasePIGNN):
    def __init__(self, input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc='hard', neumann=False,
                 muscl=False, gnn='gcn', **kwargs):
        super().__init__(input_dim, output_dim, num_layers, hidden_units, bias, act, hard_bc, gnn, **kwargs)
        self.neumann = neumann
        self.muscl = muscl


    def _hard_constraint(self, x, U, **kwargs):
        rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
        rho = 1e-3 + F.softplus(rho)
        p = 1e-3 + F.softplus(p)
        m_inf = kwargs.get('m_inf')
        d_in = 3.4 ** 2 / 9.0 - (x[:, 0:1] - 1.6 / 3.0) ** 2 - x[:, 1:2] ** 2
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

    def _face_state(self, x, rho, u, v, p, **kwargs):
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
            pos = x[:, :2]
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

    def _partial_term(self, x, U, **kwargs):
        n_idx = U.shape[0]
        edge_nodes = kwargs.get('edge_nodes')
        dual_length = kwargs.get('dual_length')
        dual_norm = kwargs.get('dual_norm')
        dual_areas = kwargs.get('dual_areas')
        fnx, fny = dual_norm[:, 0:1], dual_norm[:, 1:2]
        vi, vj = edge_nodes[:, 0], edge_nodes[:, 1]
        rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
        vars_l, vars_r = self._face_state(x, rho, u, v, p, **kwargs)
        Fa = self._riemann_flux(vars_l, vars_r, vi, vj, fnx, fny, dual_length, n_idx)
        F = Fa / dual_areas
        l1, l2, l3, l4 = F[:, 0:1], F[:, 1:2], F[:, 2:3], F[:, 3:4]
        return (l1, l2, l3, l4)

    def residual(self, U, partial_term, x, **kwargs):
        rho, u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3], U[:, 3:4]
        l1, l2, l3, l4 = partial_term
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
        return (bound_res, pde_res)
