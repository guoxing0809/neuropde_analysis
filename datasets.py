import os
import scipy
import torch
import numpy as np


class DataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self._parse_data(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _parse_data(self, data):
        self.args = data.get('args', [])
        self.num_items = self.args[0].shape[0]
        self.kwargs = data.get('kwargs', None)
        self.base_indices = np.arange(self.num_items)

    def __iter__(self):
        indices = self.base_indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, self.num_items, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch = {}
            batch['args'] = [arg[batch_idx] for arg in self.args]
            if self.kwargs is not None:
                batch['kwargs'] = {key: val[batch_idx] for key, val in self.kwargs.items()}
            yield batch

    def __len__(self):
        return (self.num_items + self.batch_size - 1) // self.batch_size


class Dataset:
    def __init__(self, simulator, *args, root='data'):
        self.simulator = simulator(*args)
        self.root = root

    def _process(self, model, saving=False, **kwargs):
        raise NotImplementedError

    def load_dataset(self, model, data_file=None, **kwargs):
        if model not in ('pinn_c', 'pinn_d', 'pignn'):
            raise ValueError(f'Unrecognized model type {model}, please select one from (pinn_c, pinn_d, pignn)!')
        if data_file is not None:
            data = torch.load(os.path.join(self.root, data_file))
        else:
            data = self._process(model=model, **kwargs)
        return data


class SineDataset(Dataset):
    def __init__(self, simulator, resolution, order, root='data'):
        super().__init__(simulator, resolution, order, root=root)

    def _process(self, model, saving=None, **kwargs):
        x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1)
        y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1)
        # bound = torch.from_numpy(self.simulator.is_bound.astype(np.float32)).unsqueeze(-1)
        bound_label = torch.from_numpy(self.simulator.is_bound.astype(bool))
        if model == 'pinn_c':
            data = {'args': [x, y],
                    'kwargs': {'bound_label': bound_label}}
        if model == 'pinn_d':
            A_coo = self.simulator.formulate_coef_matrix().tocoo()
            coef_matrix = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long(),
                                                  torch.from_numpy(A_coo.data).float(),
                                                  torch.Size(A_coo.shape))
            data = {'args': [x, y],
                    'kwargs': {'coef_matrix': coef_matrix,
                               'bound_label': bound_label}}
        if model == 'pignn':
            edge_index = torch.from_numpy(self.simulator.edge_index)
            A_coo = self.simulator.formulate_coef_matrix().tocoo()
            coef_matrix = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long(),
                                                  torch.from_numpy(A_coo.data).float(),
                                                  torch.Size(A_coo.shape))
            # data = {'x': torch.cat([x, y, bound], dim=-1),
            #         'edge_index': edge_index,
            #         'kwargs': {'coef_matrix': coef_matrix,
            #                    'bound_label': bound_label}}
            data = {'x': torch.cat([x, y], dim=-1),
                    'edge_index': edge_index,
                    'kwargs': {'coef_matrix': coef_matrix,
                               'bound_label': bound_label}}
        if saving is not None:
            torch.save(data, os.path.join(self.root, saving))
        return data


class PolynomDataset(Dataset):
    def __init__(self, simulator, resolution, order, root='data'):
        super().__init__(simulator, resolution, order, root=root)

    def _process(self, model, saving=None, **kwargs):
        x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1)
        y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1)
        # bound = torch.from_numpy(self.simulator.is_bound.astype(np.float32)).unsqueeze(-1)
        bound_label = torch.from_numpy(self.simulator.is_bound.astype(bool))
        if model == 'pinn_c':
            data = {'args': [x, y],
                    'kwargs': {'bound_label': bound_label}}
        if model == 'pinn_d':
            A_coo = self.simulator.formulate_coef_matrix().tocoo()
            coef_matrix = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long(),
                                                  torch.from_numpy(A_coo.data).float(),
                                                  torch.Size(A_coo.shape))
            data = {'args': [x, y],
                    'kwargs': {'coef_matrix': coef_matrix,
                               'bound_label': bound_label}}
        if model == 'pignn':
            edge_index = torch.from_numpy(self.simulator.edge_index)
            A_coo = self.simulator.formulate_coef_matrix().tocoo()
            coef_matrix = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long(),
                                                  torch.from_numpy(A_coo.data).float(),
                                                  torch.Size(A_coo.shape))
            # data = {'x': torch.cat([x, y, bound], dim=-1),
            #         'edge_index': edge_index,
            #         'kwargs': {'coef_matrix': coef_matrix,
            #                    'bound_label': bound_label,
            #                    'bound_value': bound_value}}
            data = {'x': torch.cat([x, y], dim=-1),
                    'edge_index': edge_index,
                    'kwargs': {'coef_matrix': coef_matrix,
                               'bound_label': bound_label}}
        if saving is not None:
            torch.save(data, os.path.join(self.root, saving))
        return data


class LiouvilleDataset(Dataset):
    def __init__(self, simulator, nr, ntheta, order, root='data'):
        super().__init__(simulator, nr, ntheta, order, root=root)

    def _process(self, model, saving=None, **kwargs):
        x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1)
        y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1)
        # bound = torch.from_numpy(self.simulator.is_bound.astype(np.float32)).unsqueeze(-1)
        bound_label = torch.from_numpy(self.simulator.is_bound.astype(bool))
        if model == 'pinn_c':
            data = {'args': [x, y],
                    'kwargs': {'bound_label': bound_label}}
        if model == 'pinn_d':
            A_coo = self.simulator.formulate_coef_matrix().tocoo()
            coef_matrix = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long(),
                                                  torch.from_numpy(A_coo.data).float(),
                                                  torch.Size(A_coo.shape))
            data = {'args': [x, y],
                    'kwargs': {'coef_matrix': coef_matrix,
                               'bound_label': bound_label}}
        if model == 'pignn':
            edge_index = torch.from_numpy(self.simulator.edge_index)
            A_coo = self.simulator.formulate_coef_matrix().tocoo()
            coef_matrix = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long(),
                                                  torch.from_numpy(A_coo.data).float(),
                                                  torch.Size(A_coo.shape))
            # data = {'x': torch.cat([x, y, bound], dim=-1),
            #         'edge_index': edge_index,
            #         'kwargs': {'coef_matrix': coef_matrix,
            #                    'bound_label': bound_label}}
            data = {'x': torch.cat([x, y], dim=-1),
                    'edge_index': edge_index,
                    'kwargs': {'coef_matrix': coef_matrix,
                               'bound_label': bound_label}}
        if saving is not None:
            torch.save(data, os.path.join(self.root, saving))
        return data


class LDFDataset(Dataset):
    def __init__(self, simulator, resolution, Re, order, root='data'):
        super().__init__(simulator, resolution, Re, order, root=root)

    def _csr2coo(self, csr_array):
        coo_array = csr_array.tocoo()
        coo_tensor = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo_array.row, coo_array.col])).long(),
                                             torch.from_numpy(coo_array.data).float(),
                                             torch.Size(coo_array.shape))
        return coo_tensor

    def _build_collocated_stencils(self, dx, dy, Re, order):
        # 2nd-order: [(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)]
        # 4th-order: [(i,j), (i+2,j), (i-2,j), (i+1,j), (i-1,j), (i,j+2), (i,j-2), (i,j+1), (i,j-1)]
        # M_v: -(pu2px2 + pu2py2) or -(pv2px2 + pv2py2)
        if order == 2:
            return {'M_gx': (np.array([0, 1, 1, 0, 0], dtype=bool),
                             np.array([1 / (2 * dx), -1 / (2 * dx)])),
                    'M_gy': (np.array([0, 0, 0, 1, 1], dtype=bool),
                             np.array([1 / (2 * dy), -1 / (2 * dy)])),
                    'M_gx_bfd': (np.array([1, 1, 0, 0, 0], dtype=bool),
                                 np.array([-1 / dx, 1 / dx])),
                    'M_gy_bfd': (np.array([1, 0, 0, 1, 0], dtype=bool),
                                 np.array([-1 / dy, 1 / dy])),
                    'M_v': (np.array([1, 1, 1, 1, 1], dtype=bool),
                            np.array(
                                [2 / (Re * dx ** 2) + 2 / (Re * dy ** 2), -1 / (Re * dx ** 2), -1 / (Re * dx ** 2),
                                 -1 / (Re * dy ** 2), -1 / (Re * dy ** 2)]))}
        if order == 4:
            return {'M_gx': (np.array([0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
                             np.array([-1 / (12 * dx), 1 / (12 * dx), 8 / (12 * dx), -8 / (12 * dx)])),
                    'M_gy': (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
                             np.array([-1 / (12 * dy), 1 / (12 * dy), 8 / (12 * dy), -8 / (12 * dy)])),
                    'M_gx_bfd': (np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
                                 np.array([-3 / (2 * dx), -1 / (2 * dx), 0, 4 / (2 * dx)])),
                    'M_gy_bfd': (np.array([1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=bool),
                                 np.array([-3 / (2 * dy), -1 / (2 * dy), 0, 4 / (2 * dy)])),
                    'M_v': (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
                            np.array(
                                [30 / (12 * Re * dx ** 2) + 30 / (12 * Re * dy ** 2), 1 / (12 * Re * dx ** 2),
                                 1 / (12 * Re * dx ** 2), -16 / (12 * Re * dx ** 2), -16 / (12 * Re * dx ** 2),
                                 1 / (12 * Re * dy ** 2), 1 / (12 * Re * dy ** 2), -16 / (12 * Re * dy ** 2),
                                 -16 / (12 * Re * dy ** 2)]))}

    def _collocated_coef_matrices(self):
        dx, dy, Re, order = self.simulator.dx, self.simulator.dy, self.simulator.Re, self.simulator.order
        mesh_idx, n_idx = self.simulator.mesh_idx, self.simulator.n_idx
        matrix_builder = self.simulator.matrix_builder
        stencil_2nd = self._build_collocated_stencils(dx, dy, Re, order=2)
        stencil_4th = self._build_collocated_stencils(dx, dy, Re, order=4)
        rows_gx, cols_gx, vals_gx = matrix_builder.coo_matrix(
            mesh_idx, order, stencil_2nd['M_gx'], stencil_4th['M_gx'], add_dirichlet=False)
        rows_gy, cols_gy, vals_gy = matrix_builder.coo_matrix(
            mesh_idx, order, stencil_2nd['M_gy'], stencil_4th['M_gy'], add_dirichlet=False)
        rows_gx_bfd, cols_gx_bfd, vals_gx_bfd = matrix_builder.coo_matrix(
            mesh_idx, order, stencil_2nd['M_gx_bfd'], stencil_4th['M_gx_bfd'], add_dirichlet=False)
        rows_gy_bfd, cols_gy_bfd, vals_gy_bfd = matrix_builder.coo_matrix(
            mesh_idx, order, stencil_2nd['M_gy_bfd'], stencil_4th['M_gy_bfd'], add_dirichlet=False)
        rows_v, cols_v, vals_v = matrix_builder.coo_matrix(
            mesh_idx, order, stencil_2nd['M_v'], stencil_4th['M_v'], add_dirichlet=False)
        M_gx = scipy.sparse.csr_matrix((vals_gx, (rows_gx, cols_gx)), shape=(n_idx, n_idx))
        M_gy = scipy.sparse.csr_matrix((vals_gy, (rows_gy, cols_gy)), shape=(n_idx, n_idx))
        M_gx_bfd = scipy.sparse.csr_matrix((vals_gx_bfd, (rows_gx_bfd, cols_gx_bfd)), shape=(n_idx, n_idx))
        M_gy_bfd = scipy.sparse.csr_matrix((vals_gy_bfd, (rows_gy_bfd, cols_gy_bfd)), shape=(n_idx, n_idx))
        M_v = scipy.sparse.csr_matrix((vals_v, (rows_v, cols_v)), shape=(n_idx, n_idx))
        return M_gx, M_gy, M_gx_bfd, M_gy_bfd, M_v

    def _process(self, model, saving=None, **kwargs):
        x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1)
        y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1)
        # bound = torch.from_numpy(self.simulator.is_bound.astype(np.float32)).unsqueeze(-1)
        mesh_idx = self.simulator.mesh_idx
        # top_idx = torch.from_numpy(self.simulator.top_idx)
        # wall_idx = torch.from_numpy(self.simulator.wall_idx)
        bound_idx = torch.from_numpy(self.simulator.bound_idx)
        origin_idx = torch.from_numpy(np.atleast_1d(mesh_idx[0, 0]))  # dirichlet of p
        dirich_idx = [bound_idx, origin_idx]
        bound_labs = torch.from_numpy(self.simulator.is_bound.astype(bool))
        if model == 'pinn_c':
            Re = torch.tensor(self.simulator.Re).float()
            data = {'args': [x, y],
                    'kwargs': {'dirich_idx': dirich_idx,
                               'bound_labs': bound_labs,
                               'Re': Re}}
        else:
            coef_matrices = self._collocated_coef_matrices()
            coef_matrices = [self._csr2coo(matrix) for matrix in coef_matrices]
            if model == 'pinn_d':
                data = {'args': [x, y],
                        'kwargs': {'coef_matrices': coef_matrices,
                                   'dirich_idx': dirich_idx,
                                   'bound_labs': bound_labs}}
            if model == 'pignn':
                edge_index = torch.from_numpy(self.simulator.edge_index)
                # data = {'x': torch.cat([x, y, bound], dim=-1),
                #         'edge_index': edge_index,
                #         'kwargs': {'coef_matrices': coef_matrices,
                #                    'dirich_idx': dirich_idx,
                #                    'bound_labs': bound_labs}}
                data = {'x': torch.cat([x, y], dim=-1),
                        'edge_index': edge_index,
                        'kwargs': {'coef_matrices': coef_matrices,
                                   'dirich_idx': dirich_idx,
                                   'bound_labs': bound_labs}}
        if saving is not None:
            torch.save(data, os.path.join(self.root, saving))
        return data


class BSFDataset(Dataset):
    def __init__(self, simulator, resolution, Re, order, root='data'):
        super().__init__(simulator, resolution, Re, order, root=root)

    def _csr2coo(self, csr_array):
        coo_array = csr_array.tocoo()
        coo_tensor = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo_array.row, coo_array.col])).long(),
                                             torch.from_numpy(coo_array.data).float(),
                                             torch.Size(coo_array.shape))
        return coo_tensor

    def _build_collocated_stencils(self, dx, dy, Re, order):
        # 2nd-order: [(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)]
        # 4th-order: [(i,j), (i+2,j), (i-2,j), (i+1,j), (i-1,j), (i,j+2), (i,j-2), (i,j+1), (i,j-1)]
        # M_v: -(pu2px2 + pu2py2) or -(pv2px2 + pv2py2)
        if order == 2:
            return {'M_gx': (np.array([0, 1, 1, 0, 0], dtype=bool),
                             np.array([1 / (2 * dx), -1 / (2 * dx)])),
                    'M_gy': (np.array([0, 0, 0, 1, 1], dtype=bool),
                             np.array([1 / (2 * dy), -1 / (2 * dy)])),
                    'M_gx_bfd': (np.array([1, 1, 0, 0, 0], dtype=bool),
                                 np.array([-1 / dx, 1 / dx])),
                    'M_gy_bfd': (np.array([1, 0, 0, 1, 0], dtype=bool),
                                 np.array([-1 / dy, 1 / dy])),
                    'M_v': (np.array([1, 1, 1, 1, 1], dtype=bool),
                            np.array(
                                [2 / (Re * dx ** 2) + 2 / (Re * dy ** 2), -1 / (Re * dx ** 2), -1 / (Re * dx ** 2),
                                 -1 / (Re * dy ** 2), -1 / (Re * dy ** 2)]))}
        if order == 4:
            return {'M_gx': (np.array([0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
                             np.array([-1 / (12 * dx), 1 / (12 * dx), 8 / (12 * dx), -8 / (12 * dx)])),
                    'M_gy': (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
                             np.array([-1 / (12 * dy), 1 / (12 * dy), 8 / (12 * dy), -8 / (12 * dy)])),
                    'M_gx_bfd': (np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
                                 np.array([-3 / (2 * dx), -1 / (2 * dx), 0, 4 / (2 * dx)])),
                    'M_gy_bfd': (np.array([1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=bool),
                                 np.array([-3 / (2 * dy), -1 / (2 * dy), 0, 4 / (2 * dy)])),
                    'M_v': (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
                            np.array(
                                [30 / (12 * Re * dx ** 2) + 30 / (12 * Re * dy ** 2), 1 / (12 * Re * dx ** 2),
                                 1 / (12 * Re * dx ** 2), -16 / (12 * Re * dx ** 2), -16 / (12 * Re * dx ** 2),
                                 1 / (12 * Re * dy ** 2), 1 / (12 * Re * dy ** 2), -16 / (12 * Re * dy ** 2),
                                 -16 / (12 * Re * dy ** 2)]))}

    def _collocated_coef_matrices(self):
        dx, dy, Re, order = self.simulator.dx, self.simulator.dy, self.simulator.Re, self.simulator.order
        nx1, nx2, ny1, ny2 = self.simulator.nx1, self.simulator.nx2, self.simulator.ny1, self.simulator.ny2
        mesh_idx1, n_idx = self.simulator.mesh_idx1, self.simulator.n_idx
        mesh_idx2 = np.vstack([mesh_idx1[ny1 - 4:, nx1 - nx2:], self.simulator.mesh_idx2])  # ny1 > 4
        matrix_builder = self.simulator.matrix_builder
        stencil_2nd = self._build_collocated_stencils(dx, dy, Re, order=2)
        stencil_4th = self._build_collocated_stencils(dx, dy, Re, order=4)
        rows_gx1, cols_gx1, vals_gx1 = matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_gx'], stencil_4th['M_gx'], add_dirichlet=False)
        rows_gy1, cols_gy1, vals_gy1 = matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_gy'], stencil_4th['M_gy'], add_dirichlet=False)
        rows_gx1_bfd, cols_gx1_bfd, vals_gx1_bfd = matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_gx_bfd'], stencil_4th['M_gx_bfd'], add_dirichlet=False)
        rows_gy1_bfd, cols_gy1_bfd, vals_gy1_bfd = matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_gy_bfd'], stencil_4th['M_gy_bfd'], add_dirichlet=False)
        rows_v1, cols_v1, vals_v1 = matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_v'], stencil_4th['M_v'], add_dirichlet=False)
        rows_gx2, cols_gx2, vals_gx2 = matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_gx'], stencil_4th['M_gx'], add_dirichlet=False)
        rows_gy2, cols_gy2, vals_gy2 = matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_gy'], stencil_4th['M_gy'], add_dirichlet=False)
        rows_gx2_bfd, cols_gx2_bfd, vals_gx2_bfd = matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_gx_bfd'], stencil_4th['M_gx_bfd'], add_dirichlet=False)
        rows_gy2_bfd, cols_gy2_bfd, vals_gy2_bfd = matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_gy_bfd'], stencil_4th['M_gy_bfd'], add_dirichlet=False)
        rows_v2, cols_v2, vals_v2 = matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_v'], stencil_4th['M_v'], add_dirichlet=False)
        # remove overlapping part
        redund_idx1 = mesh_idx1[-2, nx1 - nx2 + 1:-1]
        redund_idx2 = mesh_idx1[-3, nx1 - nx2 + 1:-1]
        mask_g1, mask_g2 = ~np.isin(rows_gx1, redund_idx1), ~np.isin(rows_gx2, redund_idx2)
        mask_g1_bfd, mask_g2_bfd = ~np.isin(rows_gx1_bfd, redund_idx1), ~np.isin(rows_gx2_bfd, redund_idx2)
        mask_v1, mask_v2 = ~np.isin(rows_v1, redund_idx1), ~np.isin(rows_v2, redund_idx2)
        rows_gx = np.hstack([rows_gx1[mask_g1], rows_gx2[mask_g2]])
        cols_gx = np.hstack([cols_gx1[mask_g1], cols_gx2[mask_g2]])
        vals_gx = np.hstack([vals_gx1[mask_g1], vals_gx2[mask_g2]])
        rows_gy = np.hstack([rows_gy1[mask_g1], rows_gy2[mask_g2]])
        cols_gy = np.hstack([cols_gy1[mask_g1], cols_gy2[mask_g2]])
        vals_gy = np.hstack([vals_gy1[mask_g1], vals_gy2[mask_g2]])
        rows_gx_bfd = np.hstack([rows_gx1_bfd[mask_g1_bfd], rows_gx2_bfd[mask_g2_bfd]])
        cols_gx_bfd = np.hstack([cols_gx1_bfd[mask_g1_bfd], cols_gx2_bfd[mask_g2_bfd]])
        vals_gx_bfd = np.hstack([vals_gx1_bfd[mask_g1_bfd], vals_gx2_bfd[mask_g2_bfd]])
        rows_gy_bfd = np.hstack([rows_gy1_bfd[mask_g1_bfd], rows_gy2_bfd[mask_g2_bfd]])
        cols_gy_bfd = np.hstack([cols_gy1_bfd[mask_g1_bfd], cols_gy2_bfd[mask_g2_bfd]])
        vals_gy_bfd = np.hstack([vals_gy1_bfd[mask_g1_bfd], vals_gy2_bfd[mask_g2_bfd]])
        rows_v = np.hstack([rows_v1[mask_v1], rows_v2[mask_v2]])
        cols_v = np.hstack([cols_v1[mask_v1], cols_v2[mask_v2]])
        vals_v = np.hstack([vals_v1[mask_v1], vals_v2[mask_v2]])
        M_gx = scipy.sparse.csr_matrix((vals_gx, (rows_gx, cols_gx)), shape=(n_idx, n_idx))
        M_gy = scipy.sparse.csr_matrix((vals_gy, (rows_gy, cols_gy)), shape=(n_idx, n_idx))
        M_gx_bfd = scipy.sparse.csr_matrix((vals_gx_bfd, (rows_gx_bfd, cols_gx_bfd)), shape=(n_idx, n_idx))
        M_gy_bfd = scipy.sparse.csr_matrix((vals_gy_bfd, (rows_gy_bfd, cols_gy_bfd)), shape=(n_idx, n_idx))
        M_v = scipy.sparse.csr_matrix((vals_v, (rows_v, cols_v)), shape=(n_idx, n_idx))
        return M_gx, M_gy, M_gx_bfd, M_gy_bfd, M_v

    def _process(self, model, saving=None, **kwargs):
        x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1)
        y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1)
        # bound = torch.from_numpy(self.simulator.is_bound.astype(np.float32)).unsqueeze(-1)
        inlet_idx = torch.from_numpy(self.simulator.inlet_idx)
        outlet_idx = torch.from_numpy(self.simulator.outlet_idx)
        inner_outlet_idx = outlet_idx - 1
        wall_idx = torch.from_numpy(self.simulator.wall_idx)
        bound_labs = torch.from_numpy(self.simulator.is_bound.astype(bool))
        if model == 'pinn_c':
            Re = torch.tensor(self.simulator.Re).float()
            data = {'args': [x, y],
                    'kwargs': {'inlet_idx': inlet_idx,
                               'outlet_idx': outlet_idx,
                               'wall_idx': wall_idx,
                               'bound_labs': bound_labs,
                               'Re': Re}}
        else:
            coef_matrices = self._collocated_coef_matrices()
            coef_matrices = [self._csr2coo(matrix) for matrix in coef_matrices]
            if model == 'pinn_d':
                data = {'args': [x, y],
                        'kwargs': {'coef_matrices': coef_matrices,
                                   'inlet_idx': inlet_idx,
                                   'outlet_idx': outlet_idx,
                                   'inner_outlet_idx': inner_outlet_idx,
                                   'wall_idx': wall_idx,
                                   'bound_labs': bound_labs}}
            if model == 'pignn':
                edge_index = torch.from_numpy(self.simulator.edge_index)
                # data = {'x': torch.cat([x, y, bound], dim=-1),
                #         'edge_index': edge_index,
                #         'kwargs': {'coef_matrices': coef_matrices,
                #                    'dirich_dist': dirich_dist,
                #                    'dirich_vals': dirich_vals,
                #                    'bound_labs': bound_labs}}
                data = {'x': torch.cat([x, y], dim=-1),
                        'edge_index': edge_index,
                        'kwargs': {'coef_matrices': coef_matrices,
                                   'inlet_idx': inlet_idx,
                                   'outlet_idx': outlet_idx,
                                   'inner_outlet_idx': inner_outlet_idx,
                                   'wall_idx': wall_idx,
                                   'bound_labs': bound_labs}}
        if saving is not None:
            torch.save(data, os.path.join(self.root, saving))
        return data


class BSWDataset(Dataset):
    """
    DNS dimensionalization:
        ρ* = ρ/ρ∞
        u*, v* = u/c∞, v/c∞
        p* = p/(ρ∞·(c∞)^2)
        p* = ρ*·T*/γ
        c* = T*^(1/2)
    """

    def __init__(self, simulator, m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y,
                 L_ref, root='data'):
        super().__init__(simulator, m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x,
                         top_y, L_ref, root=root)

    def _process(self, model, saving=None, **kwargs):
        # x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1)
        # y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1)
        x = torch.from_numpy(self.simulator.pos[:, 0].astype(np.float32)).unsqueeze(-1) / 3.0
        y = torch.from_numpy(self.simulator.pos[:, 1].astype(np.float32)).unsqueeze(-1) / 3.0
        # bound = torch.from_numpy(self.simulator.is_bound.astype(np.float32)).unsqueeze(-1)
        surf_idx = torch.from_numpy(self.simulator.surf_vidx)
        inlet_idx = torch.from_numpy(self.simulator.inlet_vidx)
        outlet_idx = torch.from_numpy(self.simulator.outlet_vidx)
        inner_surf_idx = torch.from_numpy(self.simulator.inner_surf_vidx)
        inner_outlet_idx = torch.from_numpy(self.simulator.inner_outlet_vidx)
        surf_norm = torch.from_numpy(self.simulator.surf_norm.astype(np.float32))
        m_inf = self.simulator.m_inf
        if model == 'pinn_c':
            bound_idx = np.hstack([surf_idx, inlet_idx])
            is_bound = np.isin(np.arange(self.simulator.n_idx), bound_idx)
            bound_labs = torch.from_numpy(is_bound.astype(bool))
            data = {'args': [x, y],
                    'kwargs': {'surf_idx': surf_idx,
                               'inlet_idx': inlet_idx,
                               'surf_norm': surf_norm,
                               'bound_labs': bound_labs,
                               'm_inf': torch.tensor(m_inf).float()}}
        else:
            bound_labs = torch.from_numpy(self.simulator.is_bound.astype(bool))
            lsm_neighbors = torch.from_numpy(self.simulator.lsm_neighbors)
            lsm_matrices = torch.from_numpy(self.simulator.lsm_matrices.astype(np.float32))
            edge_nodes = torch.from_numpy(self.simulator.edge_nodes.astype(np.int64))
            dual_length = torch.from_numpy(self.simulator.dual_length.astype(np.float32)).unsqueeze(-1)
            dual_norm = torch.from_numpy(self.simulator.dual_norm.astype(np.float32))
            dual_areas = torch.from_numpy(self.simulator.dual_areas.astype(np.float32)).unsqueeze(-1)
            dual_centers = torch.from_numpy(self.simulator.dual_centers.astype(np.float32)) / 3.0
            if model == 'pinn_d':
                data = {'args': [x, y],
                        'kwargs': {'edge_nodes': edge_nodes,
                                   'dual_length': dual_length,
                                   'dual_norm': dual_norm,
                                   'dual_areas': dual_areas,
                                   'dual_centers': dual_centers,
                                   'lsm_neighbors': lsm_neighbors,
                                   'lsm_matrices': lsm_matrices,
                                   'surf_idx': surf_idx,
                                   'inlet_idx': inlet_idx,
                                   'outlet_idx': outlet_idx,
                                   'surf_norm': surf_norm,
                                   'inner_surf_idx': inner_surf_idx,
                                   'inner_outlet_idx': inner_outlet_idx,
                                   'bound_labs': bound_labs,
                                   'm_inf': torch.tensor(m_inf).float()}}
            if model == 'pignn':
                edge_nodes = torch.from_numpy(self.simulator.edge_nodes.astype(np.int64))
                edge_forward = edge_nodes.T
                edge_backward = torch.flip(edge_nodes, dims=[1]).T
                edge_index = torch.cat([edge_forward, edge_backward], dim=-1)
                data = {'x': torch.cat([x, y], dim=-1),
                        'edge_index': edge_index,
                        'kwargs': {'edge_nodes': edge_nodes,
                                   'dual_length': dual_length,
                                   'dual_norm': dual_norm,
                                   'dual_areas': dual_areas,
                                   'dual_centers': dual_centers,
                                   'lsm_neighbors': lsm_neighbors,
                                   'lsm_matrices': lsm_matrices,
                                   'surf_idx': surf_idx,
                                   'inlet_idx': inlet_idx,
                                   'outlet_idx': outlet_idx,
                                   'surf_norm': surf_norm,
                                   'inner_surf_idx': inner_surf_idx,
                                   'inner_outlet_idx': inner_outlet_idx,
                                   'bound_labs': bound_labs,
                                   'm_inf': torch.tensor(m_inf).float()}}
        if saving is not None:
            torch.save(data, os.path.join(self.root, saving))
        return data


if __name__ == '__main__':
    from dns_solvers import LDFFDMSolver, BSFFDMSolver
    def test_data(data):
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                print(f'{key}: {val.shape}')
            elif isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        print(f'{k}: {v.shape}')
                    elif isinstance(v, list):
                        print(f'{k}: list of {len(v)}')
                        for i, sub_v in enumerate(v):
                            if isinstance(sub_v, torch.Tensor):
                                print(f'  [{i}]: {sub_v.shape}')
                    else:
                        print(f'{k}: {type(v)}')
            elif isinstance(val, list):
                print(f'{key}: list of {len(val)}')
                for i, sub_v in enumerate(val):
                    if isinstance(sub_v, torch.Tensor):
                        print(f'  [{i}]: {sub_v.shape}')
            else:
                print(f'{key}: {type(val)}')


    ldf_dataset = LDFDataset(LDFFDMSolver, resolution=(20, 20), Re=100, order=2)
    ldf_data = ldf_dataset.load_dataset(model='pinn_c')
    bsf_dataset = BSFDataset(BSFFDMSolver, resolution=(40, 20), Re=100, order=2)
    bsf_data = bsf_dataset.load_dataset(model='pinn_d')
    # test_data(ldf_data)
    # test_data(bsf_data)
