import os
import json
import warnings

import cupyx
import torch
import numpy as np
from scipy.sparse import coo_matrix, diags, csr_matrix, bmat
from scipy.sparse.linalg import spsolve, gmres
from dns_solvers.pdes import LinearPoisson2D, SinePoisson
from dns_solvers.pdes import NonlinearPoisson2D, PolynomPoisson
from dns_solvers.pdes import Liouville2D
from dns_solvers.pdes import LidDrivenFlow2D, BackwardStepFlow2D
from dns_solvers.pdes import BluntBowShock2D
# GPU
import cupy as cp
from cupyx.scipy.sparse.linalg import spsolve as gpu_spsolve
from cupyx.scipy.sparse.linalg import gmres as gpu_gmres
from cupyx.scipy.sparse import coo_matrix as gpu_coo_matrix
from cupyx.scipy.sparse import diags as gpu_diags
from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
from cupyx.scipy.sparse import bmat as gpu_bmat


class LinearPoissonFDMSolver:
    """
    Finite Difference Method for Regular Mesh in Cartesian Coordinates (x, y)
    Second-order Central Difference
    pu2px2 = (u_i+1,j - 2·u_i,j + u_i-1,j) / (dx^2)  →  truncation error: O(dx^2)
    pu2py2 = (u_i,j+1 - 2·u_i,j + u_i,j-1) / (dy^2)  →  truncation error: O(dy^2)
    Fourth-order Central Difference
    pu2px2 = (-u_i+2,j + 16·u_i+1,j - 30·u_i,j + 16·u_i-1,j - u_i-2,j) / (12·dx^2)  →  truncation error: O(dx^4)
    pu2py2 = (-u_i,j+2 + 16·u_i,j+1 - 30·u_i,j + 16·u_i,j-1 - u_i,j-2) / (12·dy^2)  →  truncation error: O(dy^4)
    """

    def __init__(self, order=2):
        if order not in (2, 4):
            raise ValueError(f'FDM for order={order} is not implemented!')
        self.order = order
        self.nx = None
        self.ny = None
        self.dx = None
        self.dy = None
        self.n_idx = None
        self.bound_idx = None
        self.bound_value = None
        self.internal_idx = None
        self.pos = None

    def _ij2idx(self, i, j):
        # index transform: u_ij(shape=[nx, ny])  →  u_idx(shape=[nx*ny, ])
        idx = j * self.nx + i
        return idx

    def _idx2ij(self, idx):
        # index transform: u_idx(shape=[nx*ny, ])  →  u_ij(shape=[nx, ny])
        i = idx % self.nx
        j = idx // self.nx
        return (i, j)

    def _get_nearbound(self):
        # for higher orders, reduce to 2nd-order to avoid ghost points
        grid_i, grid_j = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        is_nearbound = (((grid_i == 1) | (grid_i == self.ny - 2)) & (grid_j >= 1) & (grid_j <= self.nx - 2)) | (
                ((grid_j == 1) | (grid_j == self.nx - 2)) & (grid_i >= 1) & (grid_i <= self.ny - 2))
        is_nearbound = is_nearbound.ravel()
        nearbound_idx = np.where(is_nearbound)[0]
        return nearbound_idx

    def _stencil_length(self, order):
        if order == 2:
            return 5
        if order == 4:
            return 9

    def _stencil_idx(self, idx, order):
        if order == 2:
            return [idx, idx + 1, idx - 1, idx + self.nx, idx - self.nx]
        if order == 4:
            return [idx, idx + 2, idx - 2, idx + 1, idx - 1, idx + 2 * self.nx, idx - 2 * self.nx, idx + self.nx,
                    idx - self.nx]

    def _stencil_coefs(self, order):
        if order == 2:
            # a0·u_i,j + a1·u_i+1,j + a2·u_i-1,j + a3·u_i,j+1 + a4·u_i,j-1 = b_ij  →  [a0, a1, a2, a3, a4]
            coef_x2 = 1 / (self.dx ** 2)
            coef_y2 = 1 / (self.dy ** 2)
            return [2 * (coef_x2 + coef_y2), -coef_x2, -coef_x2, -coef_y2, -coef_y2]
        if order == 4:
            # u_i,j  u_i+2,j  u_i-2,j  u_i+1,j  u_i-1,j  u_i,j+2  u_i,j-2  u_i,j+1  u_i,j-1
            coef_x2 = 1 / (12 * self.dx ** 2)
            coef_y2 = 1 / (12 * self.dy ** 2)
            return [30 * (coef_x2 + coef_y2), coef_x2, coef_x2, -16 * coef_x2, -16 * coef_x2, coef_y2, coef_y2,
                    -16 * coef_y2,
                    -16 * coef_y2]

    def _formulate_coef_vector(self, idx, order, is_bound=False):
        # A[idx, :], shape=[n_idx, ]
        A_idx = np.zeros(self.n_idx)
        if not is_bound:
            stencil_idx = self._stencil_idx(idx, order)
            A_idx[stencil_idx] = self._stencil_coefs(order)
        else:
            A_idx[idx] = 1.
        return A_idx

    def formulate_coef_matrix(self):
        bound_idx = self.bound_idx
        internal_idx = self.internal_idx
        nearbound_idx = []
        if self.order == 4:
            nearbound_idx = self._get_nearbound()
            internal_idx = np.setdiff1d(internal_idx, nearbound_idx)
        rows = []
        cols = []
        data = []
        for idx in bound_idx:
            rows.append(idx)
            cols.append(idx)
            data.append(1.)
        for idx in internal_idx:
            rows.extend([idx] * self._stencil_length(order=self.order))
            cols.extend(self._stencil_idx(idx, order=self.order))
            data.extend(self._stencil_coefs(order=self.order))
        for idx in nearbound_idx:
            rows.extend([idx] * self._stencil_length(order=2))
            cols.extend(self._stencil_idx(idx, order=2))
            data.extend(self._stencil_coefs(order=2))
        A = coo_matrix((data, (rows, cols)), shape=(self.n_idx, self.n_idx)).tocsr()
        return A

    def formulate_rhs(self):
        x_idx, y_idx = self.pos[:, 0], self.pos[:, 1]
        b = self._right_term(x_idx, y_idx)
        b[self.bound_idx] = 0.0  # boundary condition
        return b

    def solve(self, in_idx=False):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        A = self.formulate_coef_matrix()
        b = self.formulate_rhs()
        u_idx = spsolve(A, b)
        if in_idx:
            return u_idx
        else:
            u_ij = u_idx.reshape(self.ny, self.nx)
            return u_ij


class NonlearPoissonFDMSolver:
    """
        Finite Difference Method for Regular Mesh in Cartesian Coordinates (x, y)
        Second-order Central Difference
        pu2px2 = (u_i+1,j - 2·u_i,j + u_i-1,j) / (dx^2)  →  truncation error: O(dx^2)
        pu2py2 = (u_i,j+1 - 2·u_i,j + u_i,j-1) / (dy^2)  →  truncation error: O(dy^2)
        Fourth-order Central Difference
        pu2px2 = (-u_i+2,j + 16·u_i+1,j - 30·u_i,j + 16·u_i-1,j - u_i-2,j) / (12·dx^2)  →  truncation error: O(dx^4)
        pu2py2 = (-u_i,j+2 + 16·u_i,j+1 - 30·u_i,j + 16·u_i,j-1 - u_i,j-2) / (12·dy^2)  →  truncation error: O(dy^4)
        """

    def __init__(self, order, iter_method='newton', init_method='gauss', max_iter=999, tol=1e-8):
        if order not in (2, 4):
            raise ValueError(f'FDM for order={order} is not implemented!')
        if iter_method not in ('newton', 'halley'):
            raise ValueError(f'Unrecognized iteration method {iter_method}, please select one from (newton, halley)!')
        if init_method not in ('zero', 'gauss'):
            raise ValueError(f'Unrecognized initialization method {init_method}, please select one from (zero, gauss)!')
        self.order = order
        self.iter_method = iter_method
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.nx = None
        self.ny = None
        self.dx = None
        self.dy = None
        self.n_idx = None
        self.bound_idx = None
        self.bound_value = None
        self.internal_idx = None
        self.pos = None
        self.coef_matrix = None

    def _initialize(self):
        if self.init_method == 'zero':
            u0 = np.zeros(self.n_idx)
        if self.init_method == 'gauss':
            u0 = np.random.randn(self.n_idx)
        u0[self.bound_idx] = self.bound_value
        return u0

    def _get_nearbound(self):
        # for higher orders, reduce to 2nd-order to avoid ghost points
        grid_i, grid_j = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        is_nearbound = (((grid_i == 1) | (grid_i == self.ny - 2)) & (grid_j >= 1) & (grid_j <= self.nx - 2)) | (
                ((grid_j == 1) | (grid_j == self.nx - 2)) & (grid_i >= 1) & (grid_i <= self.ny - 2))
        is_nearbound = is_nearbound.ravel()
        nearbound_idx = np.where(is_nearbound)[0]
        return nearbound_idx

    def _stencil_length(self, order):
        if order == 2:
            return 5
        if order == 4:
            return 9

    def _stencil_idx(self, idx, order):
        if order == 2:
            return [idx, idx + 1, idx - 1, idx + self.nx, idx - self.nx]
        if order == 4:
            return [idx, idx + 2, idx - 2, idx + 1, idx - 1, idx + 2 * self.nx, idx - 2 * self.nx, idx + self.nx,
                    idx - self.nx]

    def _stencil_coefs(self, order):
        if order == 2:
            # a0·u_i,j + a1·u_i+1,j + a2·u_i-1,j + a3·u_i,j+1 + a4·u_i,j-1 = b_ij  →  [a0, a1, a2, a3, a4]
            coef_x2 = 1 / (self.dx ** 2)
            coef_y2 = 1 / (self.dy ** 2)
            return [2 * (coef_x2 + coef_y2), -coef_x2, -coef_x2, -coef_y2, -coef_y2]
        if order == 4:
            # u_i,j  u_i+2,j  u_i-2,j  u_i+1,j  u_i-1,j  u_i,j+2  u_i,j-2  u_i,j+1  u_i,j-1
            coef_x2 = 1 / (12 * self.dx ** 2)
            coef_y2 = 1 / (12 * self.dy ** 2)
            return [30 * (coef_x2 + coef_y2), coef_x2, coef_x2, -16 * coef_x2, -16 * coef_x2, coef_y2, coef_y2,
                    -16 * coef_y2,
                    -16 * coef_y2]

    def formulate_coef_matrix(self):
        bound_idx = self.bound_idx
        internal_idx = self.internal_idx
        nearbound_idx = []
        if self.order == 4:
            nearbound_idx = self._get_nearbound()
            internal_idx = np.setdiff1d(internal_idx, nearbound_idx)
        rows = []
        cols = []
        data = []
        for idx in bound_idx:
            rows.append(idx)
            cols.append(idx)
            data.append(1.)
        for idx in internal_idx:
            rows.extend([idx] * self._stencil_length(order=self.order))
            cols.extend(self._stencil_idx(idx, order=self.order))
            data.extend(self._stencil_coefs(order=self.order))
        for idx in nearbound_idx:
            rows.extend([idx] * self._stencil_length(order=2))
            cols.extend(self._stencil_idx(idx, order=2))
            data.extend(self._stencil_coefs(order=2))
        A = coo_matrix((data, (rows, cols)), shape=(self.n_idx, self.n_idx)).tocsr()
        return A

    def _newton_iter_matrix(self, u):
        pfpu = self._right_partial_term_1st(u)
        pfpu[self.bound_idx] = 0.
        pfpu = diags(pfpu).T
        jacobian = self.coef_matrix - pfpu
        return jacobian

    def _halley_iter_matrix(self, u, newton_B, newton_du):
        pf2pu2 = self._right_partial_term_2nd(u)
        pf2pu2[self.bound_idx] = 0.
        pf2pu2 = diags(pf2pu2 * newton_du).T
        jacobian = newton_B - pf2pu2 / 2
        return jacobian

    def _iter_rhs(self, u):
        au = self.coef_matrix @ u
        fu = self._right_term(u)
        b = au - fu
        b[self.bound_idx] = 0.
        return b

    def _get_du(self, jac, b):
        du = spsolve(jac, b)
        return du

    def solve(self, alpha=1.0):
        """
        Newton 2nd-order nonlinear iteration
        g(u) = A·u - f(u) = g(u_n) + g'(u_n)·(u - u_n) + O(u-u_n) = 0
        u_n+1 = u_n - [g'(u_n)^(-1)]·g(u_n)
        [g'(u_n)^(-1)]·(u_n+1 - u_n) = -g(u_n)
        Halley 3th-order nonlinear iteration
        g(u) = A·u - f(u) = g(u_n) + g'(u_n)·(u - u_n) + 2^(-1)·g"(u_n)·(u - u_n)^(2) + O((u-u_n)^2) = 0
        u_n+1 = u_n - 2·[(2·g'(u_n)^(2) - g"(u_n)·g(u_n))^(-1)]·g'(u_n)·g(u_n)
        [g'(u_n) - g"(u_n)·du_newton]·(u_n+1 - u_n) = -g(u_n)
        """
        i = 0
        u0 = self._initialize()
        while i < self.max_iter:
            if self.iter_method == 'newton':
                B0 = self._newton_iter_matrix(u0)
                b0 = self._iter_rhs(u0)
                du = self._get_du(B0, b0)
            elif self.iter_method == 'halley':
                newton_B0 = self._newton_iter_matrix(u0)
                b0 = self._iter_rhs(u0)
                newton_du = self._get_du(newton_B0, b0)
                halley_B0 = self._halley_iter_matrix(u0, newton_B0, newton_du)
                du = self._get_du(halley_B0, b0)
            else:
                raise NotImplementedError
            u1 = u0 - alpha * du
            error_inf = np.linalg.norm(u1 - u0, ord=np.inf)
            print(f'Iteration status: iter={i}, error_inf={error_inf}')
            if error_inf <= self.tol:
                print(f'Converged to solution after {i + 1} iterations!')
                return u1
            i += 1
            u0 = u1.copy()
        print(f'No convergence at max iterations of {self.max_iter}!')
        return u1


class SinePoissonFDMSolver(LinearPoissonFDMSolver, SinePoisson):
    def __init__(self, resolution, order=2):
        LinearPoissonFDMSolver.__init__(self, order)
        SinePoisson.__init__(self, resolution)


class PolynomPoissonFDMSolver(NonlearPoissonFDMSolver, PolynomPoisson):
    def __init__(self, resoluton, order=2, iter_method='newton', init_method='gauss', max_iter=999, tol=1e-8):
        NonlearPoissonFDMSolver.__init__(self, order, iter_method, init_method, max_iter, tol)
        PolynomPoisson.__init__(self, resoluton)
        self.coef_matrix = self.formulate_coef_matrix()


class LiouvilleFDMSolver(Liouville2D):
    """
    Finite Difference Method for Regular Mesh in Polar Coordinates (r, θ)
    Equation transformation: pu2px2 + pu2py2 = pu2pr2 + pupr/r + pu2ptheta2/(r^2), r = (x^2 + y^2)^(1/2)
    Second-order Central Difference
    pu2pr2 = (u_i+1,j - 2·u_i,j + u_i-1,j) / (dr^2)  →  truncation error: O(dr^2)
    pupr = (u_i+1,j - u_i-1,j) / (2·dr)
    pu2ptheta2 = (u_i,j+1 - 2·u_i,j + u_i,j-1) / (dtheta^2)  →  truncation error: O(dtheta^2)
    Fourth-order Central Difference
    pu2pr2 = (-u_i+2,j + 16·u_i+1,j - 30·u_i,j + 16·u_i-1,j - u_i-2,j) / (12·dr^2)  →  truncation error: O(dr^4)
    pupr = (-u_i+2,j + 8·u_i+1,j - 8·u_i-1,j + u_i-2,j) / (12·dr)
    pu2ptheta2 = (-u_i,j+2 + 16·u_i,j+1 - 30·u_i,j + 16·u_i,j-1 - u_i,j-2) / (12·dtheta^2)  →  truncation error: O(dtheta^4)
    """

    def __init__(self, nr, ntheta, order=2, iter_method='newton', init_method='gauss', max_iter=999, tol=1e-8):
        super().__init__(nr, ntheta)
        if order not in (2, 4):
            raise ValueError(f'FDM for order={order} is not implemented!')
        if iter_method not in ('newton', 'halley'):
            raise ValueError(f'Unrecognized iteration method {iter_method}, please select one from (newton, halley)!')
        if init_method not in ('zero', 'gauss'):
            raise ValueError(f'Unrecognized initialization method {init_method}, please select one from (zero, gauss)!')
        self.order = order
        self.iter_method = iter_method
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.pos = self.cartes_pos
        self.coef_matrix = self.formulate_coef_matrix()

    def _initialize(self):
        if self.init_method == 'zero':
            u0 = np.zeros(self.n_idx)
        if self.init_method == 'gauss':
            u0 = np.random.randn(self.n_idx)
        u0[self.is_bound] = 0.0
        return u0

    def _get_nearbound(self):
        # for higher orders, reduce to 2nd-order to avoid ghost points
        nearbound_idx = list(range(self.nr - 1, self.n_idx, self.nr))  # second outer lap
        nearcenter_idx = list(range(2, self.n_idx, self.nr))  # second inner lap
        return nearbound_idx, nearcenter_idx

    def _stencil_length(self, order):
        if order == 2:
            return 5
        if order == 4:
            return 9

    def _stencil_idx(self, idx, order, near_center=False):
        i = (idx - 1) % self.nr
        j = (idx - 1) // self.nr
        if order == 2:
            # up1_id, down1_id = self._circle_offset1(idx)
            j_up1, j_down1 = (j + 1) % self.ntheta, (j - 1) % self.ntheta
            up1_id, down1_id = j_up1 * self.nr + i + 1, j_down1 * self.nr + i + 1
            if near_center:
                return [idx, idx + 1, 0, up1_id, down1_id]
            else:
                return [idx, idx + 1, idx - 1, up1_id, down1_id]
        if order == 4:
            # up1_id, down1_id = self._circle_offset1(idx)
            # up2_id, down2_id = self._circle_offset2(idx)
            j_up1, j_down1 = (j + 1) % self.ntheta, (j - 1) % self.ntheta
            j_up2, j_down2 = (j + 2) % self.ntheta, (j - 2) % self.ntheta
            up1_id, down1_id = j_up1 * self.nr + i + 1, j_down1 * self.nr + i + 1
            up2_id, down2_id = j_up2 * self.nr + i + 1, j_down2 * self.nr + i + 1
            if near_center:
                return [idx, idx + 2, 0, idx + 1, idx - 1, up2_id, down2_id, up1_id, down1_id]
            else:
                return [idx, idx + 2, idx - 2, idx + 1, idx - 1, up2_id, down2_id, up1_id, down1_id]

    def _stencil_coefs(self, idx, order):
        r = self.polar_pos[idx, 0]
        if order == 2:
            # a0·u_i,j + a1·u_i+1,j + a2·u_i-1,j + a3·u_i,j+1 + a4·u_i,j-1 = b_ij  →  [a0, a1, a2, a3, a4]
            coef_r = 1 / (2 * self.dr * r)
            coef_r2 = 1 / (self.dr ** 2)
            coef_theta2 = 1 / (self.dtheta ** 2 * r ** 2)
            return [-2 * (coef_r2 + coef_theta2), coef_r2 + coef_r, coef_r2 - coef_r, coef_theta2, coef_theta2]
        if order == 4:
            # u_i,j  u_i+2,j  u_i-2,j  u_i+1,j  u_i-1,j  u_i,j+2  u_i,j-2  u_i,j+1  u_i,j-1
            coef_r = 1 / (12 * self.dr * r)
            coef_r2 = 1 / (12 * self.dr ** 2)
            coef_theta2 = 1 / (12 * self.dtheta ** 2 * r ** 2)
            return [-30 * (coef_r2 + coef_theta2), -(coef_r2 + coef_r), coef_r - coef_r2, 16 * coef_r2 + 8 * coef_r,
                    16 * coef_r2 - 8 * coef_r, -coef_theta2, -coef_theta2, 16 * coef_theta2, 16 * coef_theta2]

    def formulate_coef_matrix(self):
        center_idx = self.center_idx
        centerlap_idx = self.centerlap_idx
        bound_idx = self.bound_idx
        internal_idx = self.internal_idx
        nearbound_idx, nearcenter_idx = [], []
        if self.order == 4:
            nearbound_idx, nearcenter_idx = self._get_nearbound()
            internal_idx = np.setdiff1d(internal_idx, nearbound_idx + nearcenter_idx)
        rows = []
        cols = []
        data = []
        for idx in center_idx:
            n_lap = len(centerlap_idx)
            rows.extend([idx] * (n_lap + 1))
            cols.extend([idx] + centerlap_idx)
            data.extend([-4 / (self.dr ** 2)] + [4 / (self.dr ** 2 * n_lap)] * n_lap)
        for idx in centerlap_idx:
            rows.extend([idx] * self._stencil_length(order=2))
            cols.extend(self._stencil_idx(idx, order=2, near_center=True))
            data.extend(self._stencil_coefs(idx, order=2))
        for idx in bound_idx:
            rows.append(idx)
            cols.append(idx)
            data.append(1.)
        for idx in internal_idx:
            rows.extend([idx] * self._stencil_length(order=self.order))
            cols.extend(self._stencil_idx(idx, order=self.order, near_center=False))
            data.extend(self._stencil_coefs(idx, order=self.order))
        for idx in nearbound_idx:
            rows.extend([idx] * self._stencil_length(order=2))
            cols.extend(self._stencil_idx(idx, order=2, near_center=False))
            data.extend(self._stencil_coefs(idx, order=2))
        for idx in nearcenter_idx:
            rows.extend([idx] * self._stencil_length(order=self.order))
            cols.extend(self._stencil_idx(idx, order=self.order, near_center=True))
            data.extend(self._stencil_coefs(idx, order=self.order))
        A = coo_matrix((data, (rows, cols)), shape=(self.n_idx, self.n_idx)).tocsr()
        return A

    def _newton_iter_matrix(self, u):
        pfpu = self._right_partial_term_1st(u)
        pfpu[self.is_bound] = 0.
        pfpu = diags(pfpu).T
        jacobian = self.coef_matrix - pfpu
        return jacobian

    def _halley_iter_matrix(self, u, newton_B, newton_du):
        pf2pu2 = self._right_partial_term_2nd(u)
        pf2pu2[self.is_bound] = 0.
        pf2pu2 = diags(pf2pu2 * newton_du).T
        jacobian = newton_B - pf2pu2 / 2
        return jacobian

    def _iter_rhs(self, u):
        au = self.coef_matrix @ u
        fu = self._right_term(u)
        b = au - fu
        b[self.is_bound] = 0.
        return b

    def _get_du(self, jac, b):
        du = spsolve(jac, b)
        return du

    def solve(self, alpha=1.0):
        """
        Newton 2nd-order nonlinear iteration
        g(u) = A·u - f(u) = g(u_n) + g'(u_n)·(u - u_n) + O(u-u_n) = 0
        u_n+1 = u_n - [g'(u_n)^(-1)]·g(u_n)
        [g'(u_n)^(-1)]·(u_n+1 - u_n) = -g(u_n)
        Halley 3th-order nonlinear iteration
        g(u) = A·u - f(u) = g(u_n) + g'(u_n)·(u - u_n) + 2^(-1)·g"(u_n)·(u - u_n)^(2) + O((u-u_n)^2) = 0
        u_n+1 = u_n - 2·[(2·g'(u_n)^(2) - g"(u_n)·g(u_n))^(-1)]·g'(u_n)·g(u_n)
        [g'(u_n) - g"(u_n)·du_newton]·(u_n+1 - u_n) = -g(u_n)
        """
        i = 0
        u0 = self._initialize()
        while i < self.max_iter:
            if self.iter_method == 'newton':
                B0 = self._newton_iter_matrix(u0)
                b0 = self._iter_rhs(u0)
                du = self._get_du(B0, b0)
            elif self.iter_method == 'halley':
                newton_B0 = self._newton_iter_matrix(u0)
                b0 = self._iter_rhs(u0)
                newton_du = self._get_du(newton_B0, b0)
                halley_B0 = self._halley_iter_matrix(u0, newton_B0, newton_du)
                du = self._get_du(halley_B0, b0)
            else:
                raise NotImplementedError
            u1 = u0 - alpha * du
            error_inf = np.linalg.norm(u1 - u0, ord=np.inf)
            print(f'Iteration status: iter={i}, error_inf={error_inf}')
            if error_inf <= self.tol:
                print(f'Converged to solution after {i + 1} iterations!')
                return u1
            i += 1
            u0 = u1.copy()
        print(f'No convergence at max iterations of {self.max_iter}!')
        return u1


class SINSFDMSolver:
    """
    Finite Difference Method for Steady Incompressible Navier-Stokes in Staggered Mesh
    U = (u, v, p)
    Equation discretization
        L1 = u·pupx + v·pupy + pppx/rho - μ·(pu2px2 + pu2py2)/rho = 0
        L2 = u·pvpx + v·pvpy + pppy/rho - μ·(pv2px2 + pv2py2)/rho = 0
        L3 = pupx + pvpy = 0
        G(U) = G(u, v, p) = [L1, L2, L3]^T = 0
        G(U) = G(U_k) + G'(U_k)dU       Newton iteration
        G(U) = G(U_k) + G'(U_k)dU + 0.5·G"(U_k)(dU)^2 = G(U_k) + [G'(U_k) + 0.5·G"(U_k)dU]dU        Halley iteration
    In staggered mesh, for a same center cell (i,j), we have:
        u(i,j) = u(x=x_i+dx/2, y=y_i), v(i,j) = v(x=x_i, y=y_j+dy/2), p(i,j) = (x=x_i, y=y_j)
    Second-order Central Difference
        Momentum equation in the x-direction, controlled by u in (x_i+dx/2, y_j):
            pupx(i,j) = pupx(x_i+dx/2, y_j) = (u_i+1,j - u_i-1,j) / (2·dx)
            pupy(i,j) = pupx(x_i+dx/2, y_j) = (u_i,j+1 - u_i,j-1) / (2·dy)
            pu2px2(i,j) = pu2px2(x_i+dx/2, y_j) = (u_i+1,j - 2·u_i,j + u_i-1,j) / (dx^2)
            pu2py2(i,j) = pu2py2(x_i+dx/2, y_j) = (u_i,j+1 - 2·u_i,j + u_i,j-1) / (dy^2)
            pppx(i,j) = pppx(x_i+dx/2, y_j) = (p_i+1,j - p_ij) / (dx)
        Momentum equation in the x-direction, controlled by v in (x_i, y_j+dy/2):
            pvpx(i,j) = pvpx(x_i, y_j+dy/2) = (v_i+1,j - v_i-1,j) / (2·dx)
            pvpy(i,j) = pvpy(x_i, y_j+dy/2) = (v_i,j+1 - v_i,j-1) / (2·dy)
            pv2px2(i,j) = pv2px2(x_i, y_j+dy/2) = (v_i+1,j - 2·u_i,j + v_i-1,j) / (dx^2)
            pv2py2(i,j) = pv2py2(x_i, y_j+dy/2) = (v_i,j+1 - 2·u_i,j + v_i,j-1) / (dy^2)
            pppy(i,j) = pppy(x_i, y_j+dy/2) = (p_i,j+1 - p_ij) / (dx)
        Continuity equation, controlled by p in (x_i, y_j)
            pupx(i,j) = pupx(x_i, y_j) = (u_i,j - u_i-1,j) / dx
            pvpy(i,j) = pvpx(x_i, y_j) = (v_i,j - v_i-1,j) / dy
    Fourth-order Central Difference
        Momentum equation in the x-direction, controlled by u in (x_i+dx/2, y_j):
            pppx(i,j) = pppx(x_i+dx/2, y_j) = (-p_i+2,j + 27·p_i+1,j - 27·p_ij + p_i-1,j) / (24·dx)
        Momentum equation in the x-direction, controlled by v in (x_i, y_j+dy/2):
            pppy(i,j) = pppy(x_i, y_j+dy/2) = (-p_i,j+2 + 27·p_i,j+1 - 27·p_ij + p_i-1,j) / (24·dy)
        Continuity equation, controlled by p in (x_i, y_j)
            pupx(i,j) = pupx(x_i, y_j) = (-u_i+1,j + 27·u_i,j - 27·u_i-1,j + u_i-2,j) / (24·dx)
            pvpy(i,j) = pvpy(x_i, y_j) = (-v_i,j+1 + 27·v_i,j - 27·v_i,j-1 + v_i,j-2) / (24·dy)
    """

    def __init__(self, order, iter_method='newton', max_iter=999, tol=1e-8, gpu=False):
        self.order = order
        self.iter_method = iter_method
        self.max_iter = max_iter
        self.tol = tol
        self.gpu = gpu
        if order not in (2, 4):
            raise ValueError(f'FDM for order={order} is not implemented!')
        if iter_method not in ('newton', 'halley'):
            raise ValueError(f'Unrecognized iteration method {iter_method}, please select one from (newton, halley)!')
        self._setup_device()
        self.n_idx = None
        self.dirichlet_labs = None
        self.dirichlet_vals = None
        self.neumann_srcs = None
        self.neumann_tags = None
        self.coef_matrices = None
        self.upwind_matrices = None
        self.dx = None
        self.dy = None
        self.Re = None

    def _setup_device(self):
        if self.gpu and cp.cuda.runtime.getDeviceCount() > 0:
            self.xp = cp
            self.coo_matrix = gpu_coo_matrix
            self.csr_matrix = gpu_csr_matrix
            self.diags = gpu_diags
            self.bmat = gpu_bmat
            self.spolve = gpu_spsolve
            self.gmres = gpu_gmres
        else:
            self.xp = np
            self.coo_matrix = coo_matrix
            self.csr_matrix = csr_matrix
            self.diags = diags
            self.bmat = bmat
            self.spolve = spsolve
            self.gmres = gmres

    def _build_stencils(self, dx, dy, Re, order):
        # 2nd-order: [(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)]
        # 4th-order: [(i,j), (i+2,j), (i-2,j), (i+1,j), (i-1,j), (i,j+2), (i,j-2), (i,j+1), (i,j-1)]
        # M_v: -(pu2px2 + pu2py2) or -(pv2px2 + pv2py2)
        if order == 2:
            return {'M_cx': (np.array([0, 1, 1, 0, 0], dtype=bool),
                             np.array([1 / (2 * dx), -1 / (2 * dx)])),
                    'M_cy': (np.array([0, 0, 0, 1, 1], dtype=bool),
                             np.array([1 / (2 * dy), -1 / (2 * dy)])),
                    'M_cx_bfd': (np.array([1, 1, 0, 0, 0], dtype=bool),
                                 np.array([-1 / dx, 1 / dx])),
                    'M_cy_bfd': (np.array([1, 0, 0, 1, 0], dtype=bool),
                                 np.array([-1 / dy, 1 / dy])),
                    'M_gx': (np.array([1, 1, 0, 0, 0], dtype=bool),
                             np.array([-1 / dx, 1 / dx])),
                    'M_gy': (np.array([1, 0, 0, 1, 0], dtype=bool),
                             np.array([-1 / dy, 1 / dy])),
                    'M_v': (np.array([1, 1, 1, 1, 1], dtype=bool),
                            np.array(
                                [2 / (Re * dx ** 2) + 2 / (Re * dy ** 2), -1 / (Re * dx ** 2), -1 / (Re * dx ** 2),
                                 -1 / (Re * dy ** 2), -1 / (Re * dy ** 2)])),
                    'M_dx': (np.array([1, 0, 1, 0, 0], dtype=bool),
                             np.array([1 / dx, -1 / dx])),
                    'M_dy': (np.array([1, 0, 0, 0, 1], dtype=bool),
                             np.array([1 / dy, -1 / dy]))}
        if order == 4:
            return {'M_cx': (np.array([0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
                             np.array([-1 / (12 * dx), 1 / (12 * dx), 8 / (12 * dx), -8 / (12 * dx)])),
                    'M_cy': (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
                             np.array([-1 / (12 * dy), 1 / (12 * dy), 8 / (12 * dy), -8 / (12 * dy)])),
                    'M_cx_bfd': (np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
                                 np.array([-3 / (2 * dx), -1 / (2 * dx), 0, 4 / (2 * dx)])),
                    'M_cy_bfd': (np.array([1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=bool),
                                 np.array([-3 / (2 * dy), -1 / (2 * dy), 0, 4 / (2 * dy)])),
                    'M_gx': (np.array([1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=bool),
                             np.array([-27 / (24 * dx), -1 / (24 * dx), 27 / (24 * dx), 1 / (24 * dx)])),
                    'M_gy': (np.array([1, 0, 0, 0, 0, 1, 0, 1, 1], dtype=bool),
                             np.array([-27 / (24 * dy), -1 / (24 * dy), 27 / (24 * dy), 1 / (24 * dy)])),
                    'M_v': (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
                            np.array(
                                [30 / (12 * Re * dx ** 2) + 30 / (12 * Re * dy ** 2), 1 / (12 * Re * dx ** 2),
                                 1 / (12 * Re * dx ** 2), -16 / (12 * Re * dx ** 2), -16 / (12 * Re * dx ** 2),
                                 1 / (12 * Re * dy ** 2), 1 / (12 * Re * dy ** 2), -16 / (12 * Re * dy ** 2),
                                 -16 / (12 * Re * dy ** 2)])),
                    'M_dx': (np.array([1, 0, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
                             np.array([27 / (24 * dx), 1 / (24 * dx), -1 / (24 * dx), -27 / (24 * dx)])),
                    'M_dy': (np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool),
                             np.array([27 / (24 * dy), 1 / (24 * dy), -1 / (24 * dy), -27 / (24 * dy)]))}

    def _initialize(self, init_file=None):
        if self.gpu:
            self.dirichlet_labs = utils.array2gpu(self.dirichlet_labs)
            self.dirichlet_vals = utils.array2gpu(self.dirichlet_vals)
            self.neumann_srcs = utils.array2gpu(self.neumann_srcs)
            self.neumann_tags = utils.array2gpu(self.neumann_tags)
            self.coef_matrices = utils.array2gpu(self.coef_matrices)
        if init_file is not None:
            U0 = self.xp.load(init_file)
            u0, v0, p0 = U0[:, 0], U0[:, 1], U0[:, 2]
        else:
            u0 = self.xp.zeros(self.n_idx)
            v0 = self.xp.zeros(self.n_idx)
            p0 = self.xp.zeros(self.n_idx)
        # u0[self.dirichlet_labs['u']] = self.dirichlet_vals['u']
        # v0[self.dirichlet_labs['v']] = self.dirichlet_vals['v']
        # p0[self.dirichlet_labs['p']] = self.dirichlet_vals['p']
        # p0[self.neumann_srcs['p']] = p0[self.neumann_tags['p']]
        U = [u0, v0, p0]
        keys = ['u', 'v', 'p']
        for i, key in enumerate(keys):
            lab = self.dirichlet_labs[key]
            if lab is not None:
                U[i][lab] = self.dirichlet_vals[key]
            src = self.neumann_srcs[key]
            if src is not None:
                tag = self.neumann_tags[key]
                U[i][src] = U[i][tag]
        return u0, v0, p0

    def formulate_coef_matrices(self, order):
        raise NotImplementedError

    def _upwind_matrices(self, u, v):
        M_cx, M_cy, M_cx_bfd, M_cy_bfd, *_ = self.coef_matrices
        if not hasattr(self, '_rows_map'):
            self._rows_map = self.xp.searchsorted(
                M_cx.indptr, self.xp.arange(M_cx.nnz, dtype='int32'), side='right') - 1
        # u_bfd, v_bfd = u < 0, v < 0
        u_bfd = (u < 0) & (self.xp.abs(u) * self.dx * self.Re > 2.)
        v_bfd = (v < 0) & (self.xp.abs(v) * self.dy * self.Re > 2.)
        u_mask = u_bfd[self._rows_map]
        v_mask = v_bfd[self._rows_map]
        M_cx_eff = M_cx.copy()
        M_cy_eff = M_cy.copy()
        M_cx_eff.data[u_mask] = M_cx_bfd.data[u_mask]
        M_cy_eff.data[v_mask] = M_cy_bfd.data[v_mask]
        self.upwind_matrices = (M_cx_eff, M_cy_eff)
        return self.upwind_matrices

    def _newton_iter_matrix(self, u, v, upwind, jac=True):
        M_cx, M_cy, _, _, M_gx, M_gy, M_vu, M_vv, M_dx, M_dy, M_p = self.coef_matrices
        if upwind:
            M_cx, M_cy = self._upwind_matrices(u, v)
        pl1pu = (self.diags(M_cx @ u, shape=(self.n_idx, self.n_idx), format='csr') +
                 self.diags(u, shape=(self.n_idx, self.n_idx), format='csr') @ M_cx +
                 self.diags(v, shape=(self.n_idx, self.n_idx), format='csr') @ M_cy + M_vu)
        pl1pv = self.diags(M_cy @ u, shape=(self.n_idx, self.n_idx), format='csr')
        pl1pp = M_gx
        pl2pu = self.diags(M_cx @ v, shape=(self.n_idx, self.n_idx), format='csr')
        pl2pv = (self.diags(u, shape=(self.n_idx, self.n_idx), format='csr') @ M_cx +
                 self.diags(M_cy @ v, shape=(self.n_idx, self.n_idx), format='csr') +
                 self.diags(v, shape=(self.n_idx, self.n_idx), format='csr') @ M_cy + M_vv)
        pl2pp = M_gy
        pl3pu = M_dx
        pl3pv = M_dy
        pl3pp = M_p
        block = [[pl1pu, pl1pv, pl1pp],
                 [pl2pu, pl2pv, pl2pp],
                 [pl3pu, pl3pv, pl3pp]]
        if jac:
            return self.bmat(block, format='csr')
        else:
            return block

    def _halley_iter_matrix(self, newton_block, newton_du, newton_dv, upwind):
        block = [row[:] for row in newton_block]
        M_cx, M_cy, *_ = self.coef_matrices
        if upwind:
            M_cx, M_cy = self.upwind_matrices
        block[0][0] = block[0][0] + 0.5 * (
                self.diags(M_cx @ newton_du, shape=(self.n_idx, self.n_idx), format='csr') +
                self.diags(newton_du, shape=(self.n_idx, self.n_idx), format='csr') @ M_cx +
                self.diags(newton_dv, shape=(self.n_idx, self.n_idx), format='csr') @ M_cy)
        block[0][1] = block[0][1] + 0.5 * self.diags(M_cy @ newton_du, shape=(self.n_idx, self.n_idx), format='csr')
        block[1][0] = block[1][0] + 0.5 * self.diags(M_cx @ newton_dv, shape=(self.n_idx, self.n_idx), format='csr')
        block[1][1] = block[1][1] + 0.5 * (
                self.diags(newton_du, shape=(self.n_idx, self.n_idx), format='csr') @ M_cx +
                self.diags(M_cy @ newton_dv, shape=(self.n_idx, self.n_idx), format='csr') +
                self.diags(newton_dv, shape=(self.n_idx, self.n_idx), format='csr') @ M_cy)
        return self.bmat(block, format='csr')

    def _iter_rhs(self, u, v, p, upwind):
        M_cx, M_cy, _, _, M_gx, M_gy, M_vu, M_vv, M_dx, M_dy, M_p = self.coef_matrices
        if upwind:
            M_cx, M_cy = self.upwind_matrices
        l1 = u * (M_cx @ u) + v * (M_cy @ u) + (M_vu @ u) + (M_gx @ p)
        l2 = u * (M_cx @ v) + v * (M_cy @ v) + (M_vv @ v) + (M_gy @ p)
        l3 = M_dx @ u + M_dy @ v + M_p @ p
        l1[self.dirichlet_labs['u']] -= self.dirichlet_vals['u']
        l2[self.dirichlet_labs['v']] -= self.dirichlet_vals['v']
        l3[self.dirichlet_labs['p']] -= self.dirichlet_vals['p']
        b = self.xp.concatenate([l1.ravel(), l2.ravel(), l3.ravel()])
        return b

    def _get_du(self, jac, b, gmres_solve, **kwargs):
        if not gmres_solve:
            dU = self.spolve(jac, b)
        else:
            rtol = kwargs.get('rtol', 1e-10)
            atol = kwargs.get('atol', 0.)
            restart = kwargs.get('restart', 30)
            maxiter = kwargs.get('maxiter', 2000)
            # dU, info = self.gmres(jac, b, rtol=rtol, atol=atol, restart=restart, maxiter=maxiter)
            dU, info = self.gmres(jac, b, tol=rtol, restart=restart, maxiter=maxiter)
            if info != 0:
                if info > 0:
                    error_msg = f'GMRES failed to converge: maxiter reached (info={info})!'
                else:
                    error_msg = f'GMRES Breakdown (info={info})!'
                raise Exception(error_msg)
        N = self.n_idx if self.n_idx is not None else dU.shape[0] // 3
        du, dv, dp = dU[:N], dU[N:2 * N], dU[2 * N:]
        return du, dv, dp

    def stag_correct(self, u, v, p):
        # u[xi+dx/2, yj] → u[xi, yj]; u[xi, yj+dy/2] → u[xi, yj]
        raise NotImplementedError

    def solve(self, upwind=False, alpha=1., beta=1., gmres_solve=False, **kwargs):
        i = 0
        init_file = kwargs.get('init_file', None)
        u0, v0, p0 = self._initialize(init_file=init_file)
        while i < self.max_iter:
            if self.iter_method == 'newton':
                B0 = self._newton_iter_matrix(u0, v0, upwind, jac=True)
                b0 = self._iter_rhs(u0, v0, p0, upwind)
                du, dv, dp = self._get_du(B0, b0, gmres_solve, **kwargs)
            elif self.iter_method == 'halley':
                newton_block = self._newton_iter_matrix(u0, v0, upwind, jac=False)
                newton_B0 = self.bmat(newton_block, format='csr')
                b0 = self._iter_rhs(u0, v0, p0, upwind)
                newton_du1, newton_dv1, _ = self._get_du(newton_B0, b0, gmres_solve, **kwargs)
                del newton_B0
                halley_B0 = self._halley_iter_matrix(newton_block, newton_du1, newton_dv1, upwind)
                du, dv, dp = self._get_du(halley_B0, b0, gmres_solve, **kwargs)
            else:
                raise NotImplementedError
            u1 = u0 - alpha * du
            v1 = v0 - alpha * dv
            p1 = p0 - beta * alpha * dp
            u_max = kwargs.get('u_max', None)
            if u_max is not None:
                u1 = self.xp.minimum(u1, u_max)
            error_u = self.xp.linalg.norm(u1 - u0, ord=np.inf)
            error_v = self.xp.linalg.norm(v1 - v0, ord=np.inf)
            error_p = self.xp.linalg.norm(p1 - p0, ord=np.inf)
            error_inf = max(float(error_u), float(error_v), float(error_p))
            print(f'Iteration status: iter={i}, error_inf={error_inf}')
            if error_inf <= self.tol:
                print(f'Converged to solution after {i + 1} iterations!')
                if self.gpu:
                    u1, v1, p1 = utils.array2cpu((u1, v1, p1))
                return self.stag_correct(u1, v1, p1)
            i += 1
            u0, v0, p0 = u1, v1, p1
        print(f'No convergence at max iterations of {self.max_iter}!')
        if self.gpu:
            u1, v1, p1 = utils.array2cpu((u1, v1, p1))
        return self.stag_correct(u1, v1, p1)


class MatrixBuilder2D:

    def internal_offset(self, nx, order):
        if order == 2:
            return np.array([0, 1, -1, nx, -nx])
        if order == 4:
            return np.array([0, 2, -2, 1, -1, 2 * nx, -2 * nx, nx, -nx])

    def dirichlet_coo(self, idx):
        cows = idx.copy()
        cols = idx.copy()
        vals = np.ones_like(idx, dtype=np.float64)
        return cows, cols, vals

    def interp_coo(self, idx, offset, coefs):
        idx = np.atleast_1d(idx)
        offset = np.atleast_1d(offset)
        coefs = np.atleast_1d(coefs).astype(np.float64)
        cols = idx[:, None] + offset
        rows = np.broadcast_to(idx[:, None], cols.shape)
        vals = np.broadcast_to(coefs, cols.shape)
        return rows.ravel(), cols.ravel(), vals.ravel()

    def neumann_coo(self, src_idx, tag_idx, dh=None):
        cows = np.broadcast_to(src_idx[:, None], (src_idx.shape[0], 2))
        cols = np.column_stack([src_idx, tag_idx])
        if dh is not None:
            vals = np.broadcast_to(np.array([1 / dh, -1 / dh])[None, :], (src_idx.shape[0], 2))
        else:
            vals = np.broadcast_to(np.array([1, -1])[None, :], (src_idx.shape[0], 2))
        return cows.ravel(), cols.ravel(), vals.ravel()

    def internal_coo(self, idx, stencil, offset):
        neigh_mask, neigh_coefs = stencil
        tag_neigh = np.where(neigh_mask)[0]
        tag_offset = offset[tag_neigh]
        cows = np.broadcast_to(idx[:, None], (idx.shape[0], neigh_coefs.shape[0]))
        cols = cows + np.broadcast_to(tag_offset[None, :], (idx.shape[0], neigh_coefs.shape[0]))
        vals = np.broadcast_to(neigh_coefs[None, :], (idx.shape[0], neigh_coefs.shape[0]))
        return cows.ravel(), cols.ravel(), vals.ravel()

    def coo_matrix(self, mesh_idx, order, stencil_2nd, stencil_4th=None, add_dirichlet=True):
        """
        :param mesh_idx: global actual idx, shape=[nx, ny], not required to be in sequence
        :param order: int, 2 or 4
        :param stencil_2nd: list/tuple, 2nd-order, (neigh_mask(bool), neigh_coefs(float))
        :param stencil_4th: list/tuple, 4th-order, (neigh_mask(bool), neigh_coefs(float))
        :param add_dirichlet: if False, exclude dirichlet boundary coefs
        :return: (rows, cols, vals) of sparse coo matrix
        stencil sequence:
            2nd-order: [(i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)]
            4th-order: [(i,j), (i+2,j), (i-2,j), (i+1,j), (i-1,j), (i,j+2), (i,j-2), (i,j+1), (i,j-1)]
            example1: 2nd-order pupx
                neigh_mask = [0, 1, 1, 0, 0]
                neigh_coefs = [1/(2·dx), -1/(2·dx)]
            example2: 2nd-order pupx + pupy
                neigh_mask = [0, 1, 1, 1, 1]
                neigh_coefs = [1/(2·dx), -1/(2·dx), -1/(2·dy), -1/(2·dy)]
        """
        if order == 4 and stencil_4th is None:
            raise ValueError(f'Please provide required stencil for 4th-order difference!')
        ny, nx = mesh_idx.shape[0], mesh_idx.shape[1]
        idx_global = mesh_idx.reshape(-1)
        idx_local = np.arange(ny * nx).reshape(ny, nx)
        if order == 2:
            offset_2nd = self.internal_offset(nx, order=2)
            internal_idx = idx_local[1:-1, 1:-1].reshape(-1)
            rows_internal, cols_internal, vals_internal = self.internal_coo(internal_idx, stencil_2nd, offset_2nd)
            if not add_dirichlet:
                rows, cols = idx_global[rows_internal], idx_global[cols_internal]
                vals = vals_internal
                return rows, cols, vals
            else:
                bound_idx = np.setdiff1d(idx_local.reshape(-1), internal_idx)
                rows_bound, cols_bound, vals_bound = self.dirichlet_coo(bound_idx)
                rows = np.hstack([rows_internal, rows_bound])
                cols = np.hstack([cols_internal, cols_bound])
                vals = np.hstack([vals_internal, vals_bound])
                rows, cols = idx_global[rows], idx_global[cols]
                return rows, cols, vals
        if order == 4:
            offset_2nd = self.internal_offset(nx, order=2)
            offset_4th = self.internal_offset(nx, order=4)
            internal_idx = idx_local[2:-2, 2:-2].reshape(-1)
            nearbound_idx = np.hstack([idx_local[1, 1:-1],
                                       idx_local[ny - 2, 1:-1],
                                       idx_local[2:-2, 1],
                                       idx_local[2:-2, nx - 2]])
            rows_internal, cols_internal, vals_internal = self.internal_coo(internal_idx, stencil_4th, offset_4th)
            rows_nearbound, cols_nearbound, vals_nearbound = self.internal_coo(nearbound_idx, stencil_2nd, offset_2nd)
            if not add_dirichlet:
                rows = np.hstack([rows_internal, rows_nearbound])
                cols = np.hstack([cols_internal, cols_nearbound])
                vals = np.hstack([vals_internal, vals_nearbound])
                rows, cols = idx_global[rows], idx_global[cols]
                return rows, cols, vals
            else:
                bound_idx = np.setdiff1d(idx_local.reshape(-1), idx_local[1:-1, 1:-1].reshape(-1))
                rows_bound, cols_bound, vals_bound = self.dirichlet_coo(bound_idx)
                rows = np.hstack([rows_internal, rows_nearbound, rows_bound])
                cols = np.hstack([cols_internal, cols_nearbound, cols_bound])
                vals = np.hstack([vals_internal, vals_nearbound, vals_bound])
                rows, cols = idx_global[rows], idx_global[cols]
                return rows, cols, vals


class LDFFDMSolver(SINSFDMSolver, LidDrivenFlow2D):
    def __init__(self, resolution, Re, order, iter_method='newton', max_iter=999, tol=1e-8, gpu=False):
        SINSFDMSolver.__init__(self, order, iter_method, max_iter, tol, gpu)
        LidDrivenFlow2D.__init__(self, resolution, Re)
        self.matrix_builder = MatrixBuilder2D()
        self.coef_matrices = self.formulate_coef_matrices(order=self.order)

    def formulate_coef_matrices(self, order=None):
        order = order if order is not None else self.order
        stencil_2nd = self._build_stencils(self.dx, self.dy, self.Re, order=2)
        stencil_4th = self._build_stencils(self.dx, self.dy, self.Re, order=4)
        rows_cx, cols_cx, vals_cx = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_cx'], stencil_4th['M_cx'], add_dirichlet=False)
        rows_cy, cols_cy, vals_cy = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_cy'], stencil_4th['M_cy'], add_dirichlet=False)
        rows_cx_bfd, cols_cx_bfd, vals_cx_bfd = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_cx_bfd'], stencil_4th['M_cx_bfd'], add_dirichlet=False)
        rows_cy_bfd, cols_cy_bfd, vals_cy_bfd = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_cy_bfd'], stencil_4th['M_cy_bfd'], add_dirichlet=False)
        rows_gx, cols_gx, vals_gx = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_gx'], stencil_4th['M_gx'], add_dirichlet=False)
        rows_gy, cols_gy, vals_gy = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_gy'], stencil_4th['M_gy'], add_dirichlet=False)
        rows_v, cols_v, vals_v = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_v'], stencil_4th['M_v'], add_dirichlet=False)
        rows_dx, cols_dx, vals_dx = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_dx'], stencil_4th['M_dx'], add_dirichlet=False)
        rows_dy, cols_dy, vals_dy = self.matrix_builder.coo_matrix(
            self.mesh_idx, order, stencil_2nd['M_dy'], stencil_4th['M_dy'], add_dirichlet=False)
        # add dirichlet boundary
        rows_ub_tb, cols_ub_tb, vals_ub_tb = self.matrix_builder.dirichlet_coo(
            np.hstack([self.mesh_idx[self.ny - 1, 1:-1], self.mesh_idx[0, 1:-1]]))
        rows_ub_l, cols_ub_l, vals_ub_l = self.matrix_builder.interp_coo(
            self.mesh_idx[:, 0], np.array([0, 1, 2]), np.array([15 / 8, -10 / 8, 3 / 8]))
        rows_ub_r, cols_ub_r, vals_ub_r = self.matrix_builder.interp_coo(
            self.mesh_idx[:, self.nx - 1], np.array([0, -1, -2]), np.array([3 / 8, 6 / 8, -1 / 8]))
        rows_vb_lr, cols_vb_lr, vals_vb_lr = self.matrix_builder.dirichlet_coo(
            np.hstack([self.mesh_idx[1:-1, 0], self.mesh_idx[1:-1, self.nx - 1]]))
        rows_vb_t, cols_vb_t, vals_vb_t = self.matrix_builder.interp_coo(
            self.mesh_idx[self.ny - 1, :], np.array([0, -self.nx, -2 * self.nx]), np.array([3 / 8, 6 / 8, -1 / 8]))
        rows_vb_b, cols_vb_b, vals_vb_b = self.matrix_builder.interp_coo(
            self.mesh_idx[0, :], np.array([0, self.nx, 2 * self.nx]), np.array([15 / 8, -10 / 8, 3 / 8]))
        rows_pb, cols_pb, vals_pb = self.matrix_builder.dirichlet_coo(self.dirichlet_labs['p'])
        # add neumann boundary
        rows_pn1, cols_pn1, vals_pn1 = self.matrix_builder.neumann_coo(
            np.hstack([self.mesh_idx[self.ny - 1, :], self.mesh_idx[0, :]]),
            np.hstack([self.mesh_idx[self.ny - 1, :] - self.nx, self.mesh_idx[0, :] + self.nx]), dh=self.dy)
        rows_pn2, cols_pn2, vals_pn2 = self.matrix_builder.neumann_coo(
            np.hstack([self.mesh_idx[1:-1, 0], self.mesh_idx[1:-1, self.nx - 1]]),
            np.hstack([self.mesh_idx[1:-1, 0] + 1, self.mesh_idx[1:-1, self.nx - 1] - 1]))
        # add pppn=0 and pb to M_p, M_gy; add dirichlet ub to M_vu; add vb to M_vv
        rows_vu = np.hstack([rows_v, rows_ub_tb, rows_ub_l, rows_ub_r])
        cols_vu = np.hstack([cols_v, cols_ub_tb, cols_ub_l, cols_ub_r])
        vals_vu = np.hstack([vals_v, vals_ub_tb, vals_ub_l, vals_ub_r])
        rows_vv = np.hstack([rows_v, rows_vb_lr, rows_vb_t, rows_vb_b])
        cols_vv = np.hstack([cols_v, cols_vb_lr, cols_vb_t, cols_vb_b])
        vals_vv = np.hstack([vals_v, vals_vb_lr, vals_vb_t, vals_vb_b])
        rows_p = np.hstack([rows_pn1, rows_pn2, rows_pb])
        cols_p = np.hstack([cols_pn1, cols_pn2, cols_pb])
        vals_p = np.hstack([vals_pn1, vals_pn2, vals_pb])
        M_cx = csr_matrix((vals_cx, (rows_cx, cols_cx)), shape=(self.n_idx, self.n_idx))
        M_cy = csr_matrix((vals_cy, (rows_cy, cols_cy)), shape=(self.n_idx, self.n_idx))
        M_cx_bfd = csr_matrix((vals_cx_bfd, (rows_cx_bfd, cols_cx_bfd)), shape=(self.n_idx, self.n_idx))
        M_cy_bfd = csr_matrix((vals_cy_bfd, (rows_cy_bfd, cols_cy_bfd)), shape=(self.n_idx, self.n_idx))
        M_gx = csr_matrix((vals_gx, (rows_gx, cols_gx)), shape=(self.n_idx, self.n_idx))
        M_gy = csr_matrix((vals_gy, (rows_gy, cols_gy)), shape=(self.n_idx, self.n_idx))
        M_vu = csr_matrix((vals_vu, (rows_vu, cols_vu)), shape=(self.n_idx, self.n_idx))
        M_vv = csr_matrix((vals_vv, (rows_vv, cols_vv)), shape=(self.n_idx, self.n_idx))
        M_dx = csr_matrix((vals_dx, (rows_dx, cols_dx)), shape=(self.n_idx, self.n_idx))
        M_dy = csr_matrix((vals_dy, (rows_dy, cols_dy)), shape=(self.n_idx, self.n_idx))
        M_p = csr_matrix((vals_p, (rows_p, cols_p)), shape=(self.n_idx, self.n_idx))
        return (M_cx, M_cy, M_cx_bfd, M_cy_bfd, M_gx, M_gy, M_vu, M_vv, M_dx, M_dy, M_p)

    def stag_correct(self, u, v, p):
        u = u.reshape(self.ny, self.nx)
        v = v.reshape(self.ny, self.nx)
        u_c, v_c = u.copy(), v.copy()
        u_c[:, 1:-1] = (u[:, :-2] + u[:, 1:-1]) / 2
        v_c[1:-1, :] = (v[:-2, :] + v[1:-1, :]) / 2
        u_c[:, 0] = 0.
        u_c[:, -1] = 0.
        v_c[0, :] = 0.
        v_c[-1, :] = 0.
        bias = p[0]
        p = p - bias
        return u_c.ravel(), v_c.ravel(), p


class BSFFDMSolver(SINSFDMSolver, BackwardStepFlow2D):
    def __init__(self, resolution, Re, order, iter_method='newton', max_iter=999, tol=1e-8, gpu=False):
        SINSFDMSolver.__init__(self, order, iter_method, max_iter, tol, gpu)
        BackwardStepFlow2D.__init__(self, resolution, Re)
        self.matrix_builder = MatrixBuilder2D()
        self.coef_matrices = self.formulate_coef_matrices(order=self.order)

    def formulate_coef_matrices(self, order=None):
        order = order if order is not None else self.order
        stencil_2nd = self._build_stencils(self.dx, self.dy, self.Re, order=2)
        stencil_4th = self._build_stencils(self.dx, self.dy, self.Re, order=4)
        mesh_idx1 = self.mesh_idx1
        mesh_idx2 = np.vstack([self.mesh_idx1[self.ny1 - 4:, self.nx1 - self.nx2:], self.mesh_idx2])  # ny1 > 4
        rows_cx1, cols_cx1, vals_cx1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_cx'], stencil_4th['M_cx'], add_dirichlet=False)
        rows_cy1, cols_cy1, vals_cy1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_cy'], stencil_4th['M_cy'], add_dirichlet=False)
        rows_cx1_bfd, cols_cx1_bfd, vals_cx1_bfd = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_cx_bfd'], stencil_4th['M_cx_bfd'], add_dirichlet=False)
        rows_cy1_bfd, cols_cy1_bfd, vals_cy1_bfd = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_cy_bfd'], stencil_4th['M_cy_bfd'], add_dirichlet=False)
        rows_gx1, cols_gx1, vals_gx1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_gx'], stencil_4th['M_gx'], add_dirichlet=False)
        rows_gy1, cols_gy1, vals_gy1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_gy'], stencil_4th['M_gy'], add_dirichlet=False)
        rows_v1, cols_v1, vals_v1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_v'], stencil_4th['M_v'], add_dirichlet=False)
        rows_dx1, cols_dx1, vals_dx1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_dx'], stencil_4th['M_dx'], add_dirichlet=False)
        rows_dy1, cols_dy1, vals_dy1 = self.matrix_builder.coo_matrix(
            mesh_idx1, order, stencil_2nd['M_dy'], stencil_4th['M_dy'], add_dirichlet=False)
        rows_cx2, cols_cx2, vals_cx2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_cx'], stencil_4th['M_cx'], add_dirichlet=False)
        rows_cy2, cols_cy2, vals_cy2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_cy'], stencil_4th['M_cy'], add_dirichlet=False)
        rows_cx2_bfd, cols_cx2_bfd, vals_cx2_bfd = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_cx_bfd'], stencil_4th['M_cx_bfd'], add_dirichlet=False)
        rows_cy2_bfd, cols_cy2_bfd, vals_cy2_bfd = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_cy_bfd'], stencil_4th['M_cy_bfd'], add_dirichlet=False)
        rows_gx2, cols_gx2, vals_gx2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_gx'], stencil_4th['M_gx'], add_dirichlet=False)
        rows_gy2, cols_gy2, vals_gy2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_gy'], stencil_4th['M_gy'], add_dirichlet=False)
        rows_v2, cols_v2, vals_v2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_v'], stencil_4th['M_v'], add_dirichlet=False)
        rows_dx2, cols_dx2, vals_dx2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_dx'], stencil_4th['M_dx'], add_dirichlet=False)
        rows_dy2, cols_dy2, vals_dy2 = self.matrix_builder.coo_matrix(
            mesh_idx2, order, stencil_2nd['M_dy'], stencil_4th['M_dy'], add_dirichlet=False)
        # add dirichlet boundary
        rows_ub_tb, cols_ub_tb, vals_ub_tb = self.matrix_builder.dirichlet_coo(
            np.hstack(
                [self.mesh_idx1[0, 1:-1], self.mesh_idx1[-1, 1: self.nx1 - self.nx2], self.mesh_idx2[-1, 1:-1]]))
        rows_ub_l1, cols_ub_l1, vals_ub_l1 = self.matrix_builder.interp_coo(
            np.hstack([self.mesh_idx1[:, 0], self.mesh_idx2[:, 0]]), np.array([0, 1, 2]),
            np.array([15 / 8, -10 / 8, 3 / 8]))
        rows_ub_l2, cols_ub_l2, vals_ub_l2 = self.matrix_builder.interp_coo(
            np.atleast_1d(self.mesh_idx1[-1, self.nx1 - self.nx2]), np.array([0, -1, -2]),
            np.array([3 / 8, 6 / 8, -1 / 8]))
        rows_ub = np.hstack([rows_ub_tb, rows_ub_l1, rows_ub_l2])
        cols_ub = np.hstack([cols_ub_tb, cols_ub_l1, cols_ub_l2])
        vals_ub = np.hstack([vals_ub_tb, vals_ub_l1, vals_ub_l2])
        rows_vb_l, cols_vb_l, vals_vb_l = self.matrix_builder.dirichlet_coo(
            np.hstack([self.mesh_idx1[1:-1, 0], self.mesh_idx2[:-1, 0], self.mesh_idx1[-1, self.nx1 - self.nx2]]))
        rows_vb_t1, cols_vb_t1, vals_vb_t1 = self.matrix_builder.interp_coo(
            self.mesh_idx1[-1, : self.nx1 - self.nx2], np.array([0, -self.nx1, -2 * self.nx2]),
            np.array([3 / 8, 6 / 8, -1 / 8]))
        rows_vb_t2, cols_vb_t2, vals_vb_t2 = self.matrix_builder.interp_coo(
            self.mesh_idx2[-1, :-1], np.array([0, -self.nx2, -2 * self.nx2]), np.array([3 / 8, 6 / 8, -1 / 8]))
        rows_vb_b, cols_vb_b, vals_vb_b = self.matrix_builder.interp_coo(
            self.mesh_idx1[0, :-1], np.array([0, self.nx1, 2 * self.nx2]), np.array([15 / 8, -10 / 8, 3 / 8]))
        rows_vb = np.hstack([rows_vb_l, rows_vb_t1, rows_vb_t2, rows_vb_b])
        cols_vb = np.hstack([cols_vb_l, cols_vb_t1, cols_vb_t2, cols_vb_b])
        vals_vb = np.hstack([vals_vb_l, vals_vb_t1, vals_vb_t2, vals_vb_b])
        rows_pb, cols_pb, vals_pb = self.matrix_builder.dirichlet_coo(
            np.hstack([self.mesh_idx1[:, -1], self.mesh_idx2[:, -1]]))
        # add neumann boundary
        rows_un, cols_un, vals_un = self.matrix_builder.neumann_coo(
            self.neumann_srcs['u'], self.neumann_tags['u'], dh=self.dx)
        rows_vn, cols_vn, vals_vn = self.matrix_builder.neumann_coo(
            self.neumann_srcs['v'], self.neumann_tags['v'], dh=self.dx)
        # rows_vn, cols_vn, vals_vn = self.matrix_builder.dirichlet_coo(self.neumann_srcs['v'])
        rows_pn1, cols_pn1, vals_pn1 = self.matrix_builder.neumann_coo(
            np.hstack([self.mesh_idx1[:, 0], self.mesh_idx2[:-1, 0]]),
            np.hstack([self.mesh_idx1[:, 0] + 1, self.mesh_idx2[:-1, 0] + 1]), dh=self.dx)
        rows_pn2, cols_pn2, vals_pn2 = self.matrix_builder.neumann_coo(
            np.hstack(
                [self.mesh_idx1[0, 1:-1], self.mesh_idx1[-1, 1: self.nx1 - self.nx2 + 1], self.mesh_idx2[-1, :-1]]),
            np.hstack([self.mesh_idx1[0, 1:-1] + self.nx1, self.mesh_idx1[-1, 1: self.nx1 - self.nx2 + 1] - self.nx1,
                       self.mesh_idx2[-1, :-1] - self.nx2]), dh=self.dy)
        # remove overlapping part
        redund_idx1 = self.mesh_idx1[-2, self.nx1 - self.nx2 + 1:-1]
        redund_idx2 = self.mesh_idx1[-3, self.nx1 - self.nx2 + 1:-1]
        mask_c1, mask_c2 = ~np.isin(rows_cx1, redund_idx1), ~np.isin(rows_cx2, redund_idx2)
        mask_c1_bfd, mask_c2_bfd = ~np.isin(rows_cx1_bfd, redund_idx1), ~np.isin(rows_cx2_bfd, redund_idx2)
        mask_g1, mask_g2 = ~np.isin(rows_gx1, redund_idx1), ~np.isin(rows_gx2, redund_idx2)
        mask_v1, mask_v2 = ~np.isin(rows_v1, redund_idx1), ~np.isin(rows_v2, redund_idx2)
        mask_d1, mask_d2 = ~np.isin(rows_dx1, redund_idx1), ~np.isin(rows_dx2, redund_idx2)
        rows_cx = np.hstack([rows_cx1[mask_c1], rows_cx2[mask_c2]])
        cols_cx = np.hstack([cols_cx1[mask_c1], cols_cx2[mask_c2]])
        vals_cx = np.hstack([vals_cx1[mask_c1], vals_cx2[mask_c2]])
        rows_cy = np.hstack([rows_cy1[mask_c1], rows_cy2[mask_c2]])
        cols_cy = np.hstack([cols_cy1[mask_c1], cols_cy2[mask_c2]])
        vals_cy = np.hstack([vals_cy1[mask_c1], vals_cy2[mask_c2]])
        rows_cx_bfd = np.hstack([rows_cx1_bfd[mask_c1_bfd], rows_cx2_bfd[mask_c2_bfd]])
        cols_cx_bfd = np.hstack([cols_cx1_bfd[mask_c1_bfd], cols_cx2_bfd[mask_c2_bfd]])
        vals_cx_bfd = np.hstack([vals_cx1_bfd[mask_c1_bfd], vals_cx2_bfd[mask_c2_bfd]])
        rows_cy_bfd = np.hstack([rows_cy1_bfd[mask_c1_bfd], rows_cy2_bfd[mask_c2_bfd]])
        cols_cy_bfd = np.hstack([cols_cy1_bfd[mask_c1_bfd], cols_cy2_bfd[mask_c2_bfd]])
        vals_cy_bfd = np.hstack([vals_cy1_bfd[mask_c1_bfd], vals_cy2_bfd[mask_c2_bfd]])
        rows_gx = np.hstack([rows_gx1[mask_g1], rows_gx2[mask_g2]])
        cols_gx = np.hstack([cols_gx1[mask_g1], cols_gx2[mask_g2]])
        vals_gx = np.hstack([vals_gx1[mask_g1], vals_gx2[mask_g2]])
        rows_gy = np.hstack([rows_gy1[mask_g1], rows_gy2[mask_g2]])
        cols_gy = np.hstack([cols_gy1[mask_g1], cols_gy2[mask_g2]])
        vals_gy = np.hstack([vals_gy1[mask_g1], vals_gy2[mask_g2]])
        rows_vu = np.hstack([rows_v1[mask_v1], rows_v2[mask_v2], rows_ub, rows_un])
        cols_vu = np.hstack([cols_v1[mask_v1], cols_v2[mask_v2], cols_ub, cols_un])
        vals_vu = np.hstack([vals_v1[mask_v1], vals_v2[mask_v2], vals_ub, vals_un])
        rows_vv = np.hstack([rows_v1[mask_v1], rows_v2[mask_v2], rows_vb, rows_vn])
        cols_vv = np.hstack([cols_v1[mask_v1], cols_v2[mask_v2], cols_vb, cols_vn])
        vals_vv = np.hstack([vals_v1[mask_v1], vals_v2[mask_v2], vals_vb, vals_vn])
        rows_dx = np.hstack([rows_dx1[mask_d1], rows_dx2[mask_d2]])
        cols_dx = np.hstack([cols_dx1[mask_d1], cols_dx2[mask_d2]])
        vals_dx = np.hstack([vals_dx1[mask_d1], vals_dx2[mask_d2]])
        rows_dy = np.hstack([rows_dy1[mask_d1], rows_dy2[mask_d2]])
        cols_dy = np.hstack([cols_dy1[mask_d1], cols_dy2[mask_d2]])
        vals_dy = np.hstack([vals_dy1[mask_d1], vals_dy2[mask_d2]])
        rows_p = np.hstack([rows_pb, rows_pn1, rows_pn2])
        cols_p = np.hstack([cols_pb, cols_pn1, cols_pn2])
        vals_p = np.hstack([vals_pb, vals_pn1, vals_pn2])
        M_cx = csr_matrix((vals_cx, (rows_cx, cols_cx)), shape=(self.n_idx, self.n_idx))
        M_cy = csr_matrix((vals_cy, (rows_cy, cols_cy)), shape=(self.n_idx, self.n_idx))
        M_cx_bfd = csr_matrix((vals_cx_bfd, (rows_cx_bfd, cols_cx_bfd)), shape=(self.n_idx, self.n_idx))
        M_cy_bfd = csr_matrix((vals_cy_bfd, (rows_cy_bfd, cols_cy_bfd)), shape=(self.n_idx, self.n_idx))
        M_gx = csr_matrix((vals_gx, (rows_gx, cols_gx)), shape=(self.n_idx, self.n_idx))
        M_gy = csr_matrix((vals_gy, (rows_gy, cols_gy)), shape=(self.n_idx, self.n_idx))
        M_vu = csr_matrix((vals_vu, (rows_vu, cols_vu)), shape=(self.n_idx, self.n_idx))
        M_vv = csr_matrix((vals_vv, (rows_vv, cols_vv)), shape=(self.n_idx, self.n_idx))
        M_dx = csr_matrix((vals_dx, (rows_dx, cols_dx)), shape=(self.n_idx, self.n_idx))
        M_dy = csr_matrix((vals_dy, (rows_dy, cols_dy)), shape=(self.n_idx, self.n_idx))
        M_p = csr_matrix((vals_p, (rows_p, cols_p)), shape=(self.n_idx, self.n_idx))
        return (M_cx, M_cy, M_cx_bfd, M_cy_bfd, M_gx, M_gy, M_vu, M_vv, M_dx, M_dy, M_p)

    def stag_correct(self, u, v, p):
        u1 = u[:self.n_idx1].reshape(self.ny1, self.nx1)
        v1 = v[:self.n_idx1].reshape(self.ny1, self.nx1)
        u2 = u[self.n_idx1:].reshape(self.ny2, self.nx2)
        v2 = v[self.n_idx1:].reshape(self.ny2, self.nx2)
        u_c1, v_c1 = u1.copy(), v1.copy()
        u_c2, v_c2 = u2.copy(), v2.copy()
        u_c1[:, 1:] = (u_c1[:, 1:] + u_c1[:, :-1]) / 2
        v_c1[1:, :] = (v_c1[1:, :] + v_c1[:-1, :]) / 2
        u_c2[:, 1:] = (u_c2[:, 1:] + u_c2[:, :-1]) / 2
        v_c2 = (v_c2 + np.vstack([v_c1[-1, self.nx1 - self.nx2:], v_c2[:-1, :]])) / 2
        yj = np.linspace(0., 1., self.ny1)
        u_c1[:, 0] = 4 * yj * (1 - yj)
        v_c1[0, :] = 0.
        u_c2[:, 0] = 0.
        u_c = np.hstack([u_c1.ravel(), u_c2.ravel()])
        v_c = np.hstack([v_c1.ravel(), v_c2.ravel()])
        return u_c, v_c, p


class CPGNSFVMSolver:
    """
        Finite Volume Method for Steady Compressible Navier-Stokes in Unstructured Mesh
        Cell type: Vertex-centered
        Pseudo time order: 1st (Euler forward/backward difference)
        Spatial order: MUSCL 2nd
        Model type: Calorically Perfect Gas (CPG)
        Conserved Vars: U = (ρ, ρu, ρv, ρE)
        Dimensionalization:
            ρ* = ρ/ρ∞
            u*, v* = u/c∞, v/c∞
            p* = p/(ρ∞·(c∞)^2)
            T* = T/T∞
            E* = E/(c∞)^2
        Gradient limiters:
            Minmod: https://doi.org/10.1016/0021-9991(83)90136-5
            VanLeer: https://doi.org/10.1016/0021-9991(74)90019-9
            Venkatakrishnan: https://doi.org/10.1006/jcph.1995.1084
            Multi-dimensional limiting process (mlp): https://doi.org/10.1016/j.compfluid.2012.04.015
        Riemann dns_solvers:
            Rusanov: https://doi.org/10.1016/0041-5553(62)90062-9
            ASUM+ (asump): https://doi.org/10.1006/jcph.1996.0256
            ASUM+up (asumpp): https://doi.org/10.1016/j.jcp.2005.09.020
            ASUMPW+ (asumpwp): https://doi:10.1006/jcph.2001.6873
            Roe: https://doi.org/10.1006/jcph.1997.5705
                 https://doi.org/10.1007/b79761 (Chapter 11)
        """

    def __init__(self, limiter='venkata', riemann='asumpp', max_iter=5000, tol=1e-8, implicit=False, cfl=1., gpu=False):
        if limiter not in ('minmod', 'vanleer', 'venkata', 'mlp'):
            raise ValueError(f'Unrecognized flux limiter! Please select one from (minmod, vanleer, venkata, mlp)!')
        if riemann not in ('rusanov', 'asump', 'asumpp', 'asumpwp', 'roe'):
            raise ValueError(
                f'Unrecognized riemann solver! Please select one from (rusanov, asump, asumpp, asumpwp, roe)!')
        if not implicit and cfl >= 1.:
            warnings.warn(f'CFL={cfl:.2f} exceeds the theoretical stability limit of 1.0!', RuntimeWarning)
        flux_limiters = {'minmod': self._minmod_limiter,
                         'vanleer': self._vanleer_limiter,
                         'venkata': self._venkata_limiter,
                         'mlp': self._mlp_limiter}
        riemann_solvers = {'rusanov': self._rusanov_solver,
                           'asump': self._asump_solver,
                           'asumpp': self._asumpp_solver,
                           'asumpwp': self._asumpwp_solver,
                           'roe': self._roe_solver}
        self.limiter = flux_limiters[limiter]
        self.riemann = riemann_solvers[riemann]
        self.max_iter = max_iter
        self.tol = tol
        self.implicit = implicit
        self.cfl = cfl  # initial cfl number
        self.gpu = gpu
        self._setup_device()
        self._setup_constants()
        self.n_idx = None
        self.m_inf = None
        self.rho_inf = None
        self.T_inf = None
        self.T_surf = None
        self.L = None
        self.aoa = None
        self.pos = None
        self.edge_nodes = None
        self.edge_cells = None
        self.edge_length = None
        self.dual_areas = None
        self.dual_length = None
        self.dual_norm = None
        self.dual_centers = None
        self.surf_length = None
        self.surf_norm = None
        self.inlet_length = None
        self.inlet_norm = None
        self.outlet_length = None
        self.outlet_norm = None
        self.surf_vidx = None
        self.outlet_vidx = None
        self.inlet_vidx = None
        self.inner_surf_vidx = None
        self.inner_inlet_vidx = None
        self.inner_outlet_vidx = None
        self.surf_eidx = None
        self.inlet_eidx = None
        self.outlet_eidx = None
        self.lsm_neighbors = None
        self.lsm_matrices = None

    def _setup_device(self):
        if self.gpu and cp.cuda.runtime.getDeviceCount() > 0:
            self.xp = cp
            self.coo_matrix = gpu_coo_matrix
            self.csr_matrix = gpu_csr_matrix
            self.diags = gpu_diags
            self.bmat = gpu_bmat
            self.spolve = gpu_spsolve
            self.gmres = gpu_gmres
            self.scatter_add = cupyx.scatter_add
            self.scatter_max = cupyx.scatter_max
            self.scatter_min = cupyx.scatter_min

        else:
            self.xp = np
            self.coo_matrix = coo_matrix
            self.csr_matrix = csr_matrix
            self.diags = diags
            self.bmat = bmat
            self.spolve = spsolve
            self.gmres = gmres
            self.scatter_add = np.add.at
            self.scatter_max = np.maximum.at
            self.scatter_min = np.minimum.at

    def _setup_constants(self):
        self.gamma = 1.4  # specific heat ratio
        self.R_air = 287  # specific gas constant of air [J/(kg·K)]
        self.T0 = 288.15  # Sutherland reference temperature [K]
        self.mu0 = 1.789e-5  # Sutherland reference viscosity [Pa·s]
        self.C0 = 110.4  # Sutherland constant for air [K]
        self.Pr = 0.72  # Prandtl number
        self.c_p = self.R_air * self.gamma / (self.gamma - 1)  # specific heat at constant pressure [J/(kg·K)]

    def _setup_inflow(self):
        self.c_inf = np.sqrt(self.gamma * self.R_air * self.T_inf)  # inflow sound velocity [m/s]
        self.mu_inf = self.mu0 * np.power(self.T_inf / self.T0, 1.5) * (self.T0 + self.C0) / (self.T_inf + self.C0)
        self.Re_inf = self.rho_inf * self.c_inf * self.L / self.mu_inf  # inflow relative Reynolds number
        self.C_inf = self.C0 / self.T_inf

    def _initialize(self):
        if self.gpu:
            self.pos = utils.array2gpu(self.pos)
            self.edge_nodes = utils.array2gpu(self.edge_nodes)
            self.edge_cells = utils.array2gpu(self.edge_cells)
            self.edge_length = utils.array2gpu(self.edge_length)
            self.dual_areas = utils.array2gpu(self.dual_areas)
            self.dual_length = utils.array2gpu(self.dual_length)
            self.dual_norm = utils.array2gpu(self.dual_norm)
            self.dual_centers = utils.array2gpu(self.dual_centers)
            self.surf_length = utils.array2gpu(self.surf_length)
            self.surf_norm = utils.array2gpu(self.surf_norm)
            self.inlet_length = utils.array2gpu(self.inlet_length)
            self.inlet_norm = utils.array2gpu(self.inlet_norm)
            self.outlet_length = utils.array2gpu(self.outlet_length)
            self.outlet_norm = utils.array2gpu(self.outlet_norm)
            self.surf_vidx = utils.array2gpu(self.surf_vidx)
            self.inlet_vidx = utils.array2gpu(self.inlet_vidx)
            self.outlet_vidx = utils.array2gpu(self.outlet_vidx)
            self.inner_surf_vidx = utils.array2gpu(self.inner_surf_vidx)
            self.inner_inlet_vidx = utils.array2gpu(self.inner_inlet_vidx)
            self.inner_outlet_vidx = utils.array2gpu(self.inner_outlet_vidx)
            self.surf_eidx = utils.array2gpu(self.surf_eidx)
            self.inlet_eidx = utils.array2gpu(self.inlet_eidx)
            self.outlet_eidx = utils.array2gpu(self.outlet_eidx)
            self.lsm_neighbors = utils.array2gpu(self.lsm_neighbors)
            self.lsm_matrices = utils.array2gpu(self.lsm_matrices)
        self.vi, self.vj = self.edge_nodes[:, 0], self.edge_nodes[:, 1]
        self.fnx, self.fny = self.dual_norm[:, 0], self.dual_norm[:, 1]
        self.dfi, self.dfj = self.dual_centers - self.pos[self.vi], self.dual_centers - self.pos[self.vj]
        self.e_ij = (self.pos[self.vj] - self.pos[self.vi]) / self.edge_length[:, None]
        self.li, self.lj = self.xp.linalg.norm(self.dfi, axis=1), self.xp.linalg.norm(self.dfj, axis=1)
        self.wi, self.wj = self.lj / (self.li + self.lj), self.li / (self.li + self.lj)
        if self.aoa is not None:
            self.mx_inf, self.my_inf = self.m_inf * np.cos(self.aoa), self.m_inf * np.sin(self.aoa)
        else:
            self.mx_inf, self.my_inf = self.m_inf, 0.0
        rho0 = self.xp.ones(self.n_idx)
        E0 = 1.0 / (self.gamma * (self.gamma - 1)) + self.m_inf ** 2 / 2.0
        rhou0, rhov0 = rho0 * self.mx_inf, rho0 * self.my_inf
        rhoE0 = rho0 * E0
        if self.T_surf is not None:
            self.T_surf = self.T_surf / self.T_inf
            rhoE0[self.surf_vidx] = self.T_surf / (self.gamma * (self.gamma - 1)) + self.m_inf ** 2 / 2.0
        return rho0, rhou0, rhov0, rhoE0

    def _scatter_aggregate(self, indices, values, minlength, aggr):
        if aggr not in ('add', 'mean', 'max', 'min'):
            raise ValueError('Unrecognized aggregator type! Please select one from (add, mean, max, min)!')
        shape = (minlength,) if values.ndim == 1 else (minlength, values.shape[-1])
        if aggr == 'add':
            output = self.xp.zeros(shape)
            self.scatter_add(output, indices, values)
            return output
        if aggr == 'mean':
            add = self.xp.zeros(shape)
            count = self.xp.zeros(shape)
            self.scatter_add(add, indices, values)
            self.scatter_add(count, indices, self.xp.ones_like(values))
            output = add / (count + 1e-10)
            return output
        if aggr == 'max':
            output = self.xp.full(shape, -self.xp.inf)
            self.scatter_max(output, indices, values)
            return output
        if aggr == 'min':
            output = self.xp.full(shape, self.xp.inf)
            self.scatter_min(output, indices, values)
            return output

    def _slope_ratio(self, phi, grad_phi):
        del_phi = phi[self.vj] - phi[self.vi]
        del_phi_i = self.xp.sum(grad_phi[self.vi] * (self.dual_centers - self.pos[self.vi]), axis=1)
        del_phi_j = self.xp.sum(grad_phi[self.vj] * (self.pos[self.vj] - self.dual_centers), axis=1)
        theta_i = del_phi / self.xp.where(del_phi_i >= 0, del_phi_i + 1e-10, del_phi_i - 1e-10)
        theta_j = del_phi / self.xp.where(del_phi_j >= 0, del_phi_j + 1e-10, del_phi_j - 1e-10)
        return self.xp.column_stack([theta_i, theta_j])
        # theta = self.xp.minimum(theta_i, theta_j)
        # return theta

    def _minmod_limiter(self, phi, grad_phi):
        theta = self._slope_ratio(phi, grad_phi)
        theta_c = self.xp.clip(theta, 0.0, 1.0)
        # weight = self._scatter_aggregate(self.edge_nodes.ravel(), theta_c.ravel(), self.n_idx, 'min')
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(theta_c, 2).ravel(), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _vanleer_limiter(self, phi, grad_phi):
        theta = self._slope_ratio(phi, grad_phi)
        theta_c = (theta + self.xp.abs(theta)) / (1.0 + self.xp.abs(theta))
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), theta_c.ravel(), self.n_idx, 'min')
        # weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(theta_c, 2).ravel(), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _venkata_limiter(self, phi, grad_phi):
        theta = self._slope_ratio(phi, grad_phi)
        theta_c = self.xp.maximum((theta ** 2 + 2.0 * theta) / (theta ** 2 + theta + 2.0), 0.0)
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), theta_c.ravel(), self.n_idx, 'min')
        # weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(theta_c, 2).ravel(), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _mlp_limiter(self, phi, grad_phi):
        phi_max = self.xp.max(phi[self.edge_nodes], axis=1)
        phi_min = self.xp.min(phi[self.edge_nodes], axis=1)
        neigh_max = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(phi_max, 2), self.n_idx, 'max')
        neigh_min = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(phi_min, 2), self.n_idx, 'min')
        del_max_i, del_max_j = neigh_max[self.vi] - phi[self.vi], neigh_max[self.vj] - phi[self.vj]
        del_min_i, del_min_j = neigh_min[self.vi] - phi[self.vi], neigh_min[self.vj] - phi[self.vj]
        del_phi_i = self.xp.sum(grad_phi[self.vi] * (self.dual_centers - self.pos[self.vi]), axis=1)
        del_phi_j = self.xp.sum(grad_phi[self.vj] * (self.dual_centers - self.pos[self.vj]), axis=1)
        del_phi_i = self.xp.where(del_phi_i >= 0, del_phi_i + 1e-10, del_phi_i - 1e-10)
        del_phi_j = self.xp.where(del_phi_j >= 0, del_phi_j + 1e-10, del_phi_j - 1e-10)
        phi_fi = phi[self.vi] + del_phi_i
        phi_fj = phi[self.vj] + del_phi_j
        theta_i = self.xp.maximum((phi[self.vj] - phi[self.vi]) / del_phi_i, 0.0)
        theta_i = self.xp.where(phi_fi > neigh_max[self.vi], del_max_i / del_phi_i, theta_i)
        theta_i = self.xp.where(phi_fi < neigh_min[self.vi], del_min_i / del_phi_i, theta_i)
        theta_j = self.xp.maximum((phi[self.vi] - phi[self.vj]) / del_phi_j, 0.0)
        theta_j = self.xp.where(phi_fj > neigh_max[self.vj], del_max_j / del_phi_j, theta_j)
        theta_j = self.xp.where(phi_fj < neigh_min[self.vj], del_min_j / del_phi_j, theta_j)
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.hstack([theta_i, theta_j]), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _rusanov_solver(self, vars_l, vars_r, U_l, U_r):
        Fa_l, Fa_r, un_l, un_r = self._flux_advection(vars_l, vars_r, U_l, U_r)
        *_, c_l = vars_l
        *_, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        # central difference + dissipation
        lambda_max = self.xp.maximum(self.xp.abs(un_l) + c_l, self.xp.abs(un_r) + c_r)
        Fa1_f = self.dual_length * (0.5 * (Fa_l[0] + Fa_r[0]) - 0.5 * lambda_max * (rho_r - rho_l))  # mass
        Fa2_f = self.dual_length * (0.5 * (Fa_l[1] + Fa_r[1]) - 0.5 * lambda_max * (rhou_r - rhou_l))  # axis-x momentum
        Fa3_f = self.dual_length * (0.5 * (Fa_l[2] + Fa_r[2]) - 0.5 * lambda_max * (rhov_r - rhov_l))  # axis-y momentum
        Fa4_f = self.dual_length * (0.5 * (Fa_l[3] + Fa_r[3]) - 0.5 * lambda_max * (rhoE_r - rhoE_l))  # energy
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _asump_solver(self, vars_l, vars_r, U_l, U_r, alpha=3 / 16, beta=1 / 8):
        u_l, v_l, _, p_l, c_l = vars_l
        u_r, v_r, _, p_r, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        un_r = u_r * self.fnx + v_r * self.fny
        H_l, H_r = (rhoE_l + p_l) / rho_l, (rhoE_r + p_r) / rho_r
        cs_l = self.xp.sqrt(2 * (self.gamma - 1) * H_l / (self.gamma + 1))
        cs_r = self.xp.sqrt(2 * (self.gamma - 1) * H_r / (self.gamma + 1))
        cbar_l = cs_l ** 2 / (self.xp.maximum(self.xp.abs(un_l), cs_l))
        cbar_r = cs_r ** 2 / (self.xp.maximum(self.xp.abs(un_r), cs_r))
        c_f = self.xp.minimum(cbar_l, cbar_r)
        m_l, m_r = un_l / c_f, un_r / c_f
        m_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (m_l + self.xp.abs(m_l)),
                               0.25 * (m_l + 1.0) ** 2 + beta * (m_l ** 2 - 1.0) ** 2)
        m_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (m_r - self.xp.abs(m_r)),
                                -0.25 * (m_r - 1.0) ** 2 - beta * (m_r ** 2 - 1.0) ** 2)
        m_f = m_plus + m_minus
        m_f_plus = 0.5 * (m_f + self.xp.abs(m_f))
        m_f_minus = 0.5 * (m_f - self.xp.abs(m_f))
        p_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (1 + self.xp.sign(m_l)),
                               0.25 * (m_l + 1.0) ** 2 * (2.0 - m_l) + alpha * m_l * (m_l ** 2 - 1) ** 2)
        p_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (1 - self.xp.sign(m_r)),
                                0.25 * (m_r - 1.0) ** 2 * (2.0 + m_r) - alpha * m_r * (m_r ** 2 - 1) ** 2)
        p_f = p_l * p_plus + p_r * p_minus
        Fa1_f = self.dual_length * c_f * (rho_l * m_f_plus + rho_r * m_f_minus)
        Fa2_f = self.dual_length * (c_f * (rhou_l * m_f_plus + rhou_r * m_f_minus) + p_f * self.fnx)
        Fa3_f = self.dual_length * (c_f * (rhov_l * m_f_plus + rhov_r * m_f_minus) + p_f * self.fny)
        Fa4_f = self.dual_length * c_f * ((rhoE_l + p_l) * m_f_plus + (rhoE_r + p_r) * m_f_minus)
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _asumpp_solver(self, vars_l, vars_r, U_l, U_r, k=0.1, kp=0.25, ku=0.1, sigma=0.5, alpha=3 / 16,
                       beta=1 / 8):
        # k = O(1), 0 ≤ kp ≤ 1, 0 ≤ ku ≤ 1, sigma ≤ 1
        u_l, v_l, _, p_l, c_l = vars_l
        u_r, v_r, _, p_r, c_r = vars_r
        rho_l, *_, rhoE_l = U_l
        rho_r, *_, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        un_r = u_r * self.fnx + v_r * self.fny
        H_l, H_r = (rhoE_l + p_l) / rho_l, (rhoE_r + p_r) / rho_r
        cs_l = self.xp.sqrt(2 * (self.gamma - 1) * H_l / (self.gamma + 1))
        cs_r = self.xp.sqrt(2 * (self.gamma - 1) * H_r / (self.gamma + 1))
        cbar_l = cs_l ** 2 / (self.xp.maximum(self.xp.abs(un_l), cs_l))
        cbar_r = cs_r ** 2 / (self.xp.maximum(self.xp.abs(un_r), cs_r))
        c_f = self.xp.minimum(cbar_l, cbar_r)
        m_l, m_r = un_l / c_f, un_r / c_f
        m2_bar = 0.5 * (m_l ** 2 + m_r ** 2)
        # m_co = self.xp.maximum(k * self.m_inf, 0.3)
        m_co = k * self.m_inf
        m2_o = self.xp.minimum(1, self.xp.maximum(m2_bar, m_co ** 2))
        m_o = self.xp.sqrt(m2_o)
        fa = m_o * (2 - m_o)
        m1_lp, m1_lm = 0.5 * (m_l + self.xp.abs(m_l)), 0.5 * (m_l - self.xp.abs(m_l))
        m1_rp, m1_rm = 0.5 * (m_r + self.xp.abs(m_r)), 0.5 * (m_r - self.xp.abs(m_r))
        m2_lp, m2_lm = 0.25 * (m_l + 1.0) ** 2, -0.25 * (m_l - 1.0) ** 2
        m2_rp, m2_rm = 0.25 * (m_r + 1.0) ** 2, -0.25 * (m_r - 1.0) ** 2
        m_plus = self.xp.where(self.xp.abs(m_l) > 1.0, m1_lp, m2_lp * (1 - 16 * beta * m2_lm))
        m_minus = self.xp.where(self.xp.abs(m_r) > 1.0, m1_rm, m2_rm * (1 + 16 * beta * m2_rp))
        m_p = kp * self.xp.maximum(1 - sigma * m2_bar, 0.0) * (p_r - p_l) / (fa * 0.5 * (rho_l + rho_r) * c_f ** 2)
        m_f = m_plus + m_minus - m_p
        p_plus = self.xp.where(self.xp.abs(m_l) > 1.0, m1_lp / m_l,
                               m2_lp * (2 - m_l - 16 * alpha * (-4 + 5 * fa ** 2) * m_l * m2_lm))
        p_minus = self.xp.where(self.xp.abs(m_r) > 1.0, m1_rm / m_r,
                                m2_rm * (-2 - m_r + 16 * alpha * (-4 + 5 * fa ** 2) * m_r * m2_rp))
        p_u = ku * p_plus * p_minus * (rho_l + rho_r) * fa * c_f * (u_r - u_l)
        p_f = p_plus * p_l + p_minus * p_r - p_u
        rho_f = self.xp.where(m_f > 0.0, rho_l, rho_r)
        Fa1_f = self.dual_length * c_f * m_f * rho_f
        u_f = self.xp.where(Fa1_f > 0, u_l, u_r)
        v_f = self.xp.where(Fa1_f > 0, v_l, v_r)
        H_f = self.xp.where(Fa1_f > 0, H_l, H_r)
        Fa2_f = Fa1_f * u_f + self.dual_length * p_f * self.fnx
        Fa3_f = Fa1_f * v_f + self.dual_length * p_f * self.fny
        Fa4_f = Fa1_f * H_f
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _asumpwp_solver(self, vars_l, vars_r, U_l, U_r, alpha=3 / 16, beta=1 / 8):
        u_l, v_l, _, p_l, c_l = vars_l
        u_r, v_r, _, p_r, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        un_r = u_r * self.fnx + v_r * self.fny
        H_l, H_r = (rhoE_l + p_l) / rho_l, (rhoE_r + p_r) / rho_r
        # H_norm = 0.5 * (H_l + H_r - 0.5 * (u_l ** 2 + v_l ** 2 + u_r ** 2 + v_r ** 2))
        H_norm = self.xp.maximum(0.5 * (H_l + H_r - 0.5 * (u_l ** 2 + v_l ** 2 + u_r ** 2 + v_r ** 2)), 1e-10)
        c_s = self.xp.sqrt(2 * (self.gamma - 1) * H_norm / (self.gamma + 1))
        c_f = self.xp.where(0.5 * (un_l + un_r) > 0, c_s ** 2 / self.xp.maximum(self.xp.abs(un_l), c_s),
                            c_s ** 2 / self.xp.maximum(self.xp.abs(un_r), c_s))
        m_l, m_r = un_l / c_f, un_r / c_f
        m_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (m_l + self.xp.abs(m_l)),
                               0.25 * (m_l + 1.0) ** 2 + beta * (m_l ** 2 - 1.0) ** 2)
        m_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (m_r - self.xp.abs(m_r)),
                                -0.25 * (m_r - 1.0) ** 2 - beta * (m_r ** 2 - 1.0) ** 2)
        mass_f = m_plus + m_minus

        p_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (1 + self.xp.sign(m_l)),
                               0.25 * (m_l + 1.0) ** 2 * (2.0 - m_l) + alpha * m_l * (m_l ** 2 - 1) ** 2)
        p_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (1 - self.xp.sign(m_r)),
                                0.25 * (m_r - 1.0) ** 2 * (2.0 + m_r) - alpha * m_r * (m_r ** 2 - 1) ** 2)
        p_s = p_plus * p_l + p_minus * p_r
        # asumpw version (asumpw+ needs pressure states of higher-order neighbors)
        f_l = self.xp.where((p_s != 0) & (self.xp.abs(m_l) < 1), (p_l / (p_s + 1e-10) - 1), 0.0)
        f_r = self.xp.where((p_s != 0) & (self.xp.abs(m_r) < 1), (p_r / (p_s + 1e-10) - 1), 0.0)
        w = 1.0 - self.xp.minimum(p_l / p_r, p_r / p_l) ** 3
        mbar_plus = self.xp.where(mass_f >= 0, m_plus + m_minus * ((1 - w) * (1 + f_r) - f_l), m_plus * w * (1 + f_l))
        mbar_minus = self.xp.where(mass_f >= 0, m_minus * w * (1 + f_r), m_minus + m_plus * ((1 - w) * (1 + f_l) - f_r))
        Fa1_f = self.dual_length * c_f * (rho_l * mbar_plus + rho_r * mbar_minus)
        Fa2_f = self.dual_length * (c_f * (rhou_l * mbar_plus + rhou_r * mbar_minus) + p_s * self.fnx)
        Fa3_f = self.dual_length * (c_f * (rhov_l * mbar_plus + rhov_r * mbar_minus) + p_s * self.fny)
        Fa4_f = self.dual_length * c_f * ((rhoE_l + p_l) * mbar_plus + (rhoE_r + p_r) * mbar_minus)
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _roe_solver(self, vars_l, vars_r, U_l, U_r):
        Fa_l, Fa_r, *_ = self._flux_advection(vars_l, vars_r, U_l, U_r)
        u_l, v_l, _, p_l, _ = vars_l
        u_r, v_r, _, p_r, _ = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        wl_rho = self.xp.sqrt(rho_l)
        wl_rhou = wl_rho * u_l
        wl_rhov = wl_rho * v_l
        wl_rhoH = wl_rho * (rhoE_l / rho_l + p_l)
        wr_rho = self.xp.sqrt(rho_r)
        wr_rhou = wr_rho * u_r
        wr_rhov = wr_rho * v_r
        wr_rhoH = wr_rho * (rhoE_r / rho_r + p_r)
        rho_roe = wl_rho + wr_rho
        u_roe = (wl_rhou + wr_rhou) / rho_roe
        v_roe = (wl_rhov + wr_rhov) / rho_roe
        H_roe = (wl_rhoH + wr_rhoH) / rho_roe
        q2_roe = u_roe ** 2 + v_roe ** 2
        T_roe = self.xp.maximum((self.gamma - 1) * (H_roe - 0.5 * q2_roe), 1e-10)
        c_roe = self.xp.sqrt(T_roe)
        lambda1 = self.xp.abs(u_roe - c_roe)
        lambda2 = self.xp.abs(u_roe)
        lambda3 = lambda2
        lambda4 = self.xp.abs(u_roe + c_roe)
        du1 = rho_r - rho_l
        du2 = rhou_r - rhou_l
        du3 = rhov_r - rhov_l
        du4 = rhoE_r - rhoE_l
        du4_bar = du4 - (du3 - v_roe * du1) * v_roe
        alpha3 = du3 - v_roe * du1
        alpha2 = (self.gamma - 1) * (du1 * (H_roe - u_roe ** 2) + u_roe * du2 - du4_bar) / c_roe ** 2
        alpha1 = (du1 * (u_roe + c_roe) - du2 - c_roe * alpha2) / (2 * c_roe)
        alpha4 = du1 - (alpha1 + alpha2)
        rho_dp = lambda1 * alpha1 + lambda2 * alpha2 + lambda4 * alpha4
        rhou_dp = lambda1 * alpha1 * (u_roe - c_roe) + lambda2 * alpha2 * u_roe + lambda4 * alpha4 * (u_roe + c_roe)
        rhov_dp = (lambda1 * alpha1 + lambda2 * alpha2) * v_roe + lambda3 * alpha3 + lambda4 * alpha4 * v_roe
        rhoE_dp = (lambda1 * alpha1 * (H_roe - u_roe * c_roe) + 0.5 * lambda2 * alpha2 * q2_roe
                   + lambda3 * alpha3 * v_roe + lambda4 * alpha4 * (H_roe + u_roe * c_roe))
        Fa1_f = self.dual_length * (0.5 * (Fa_l[0] + Fa_r[0]) - 0.5 * rho_dp)
        Fa2_f = self.dual_length * (0.5 * (Fa_l[1] + Fa_r[1]) - 0.5 * rhou_dp)
        Fa3_f = self.dual_length * (0.5 * (Fa_l[2] + Fa_r[2]) - 0.5 * rhov_dp)
        Fa4_f = self.dual_length * (0.5 * (Fa_l[3] + Fa_r[3]) - 0.5 * rhoE_dp)
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _flux_advection(self, vars_l, vars_r, U_l, U_r):
        # raw flux without dissipation
        u_l, v_l, T_l, p_l, c_l = vars_l
        u_r, v_r, T_r, p_r, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        Fa1_l = rho_l * un_l
        Fa2_l = rho_l * un_l * u_l + p_l * self.fnx
        Fa3_l = rho_l * un_l * v_l + p_l * self.fny
        Fa4_l = (rhoE_l + p_l) * un_l
        un_r = u_r * self.fnx + v_r * self.fny
        Fa1_r = rho_r * un_r
        Fa2_r = rho_r * un_r * u_r + p_r * self.fnx
        Fa3_r = rho_r * un_r * v_r + p_r * self.fny
        Fa4_r = (rhoE_r + p_r) * un_r
        Fa_l = (Fa1_l, Fa2_l, Fa3_l, Fa4_l)
        Fa_r = (Fa1_r, Fa2_r, Fa3_r, Fa4_r)
        return Fa_l, Fa_r, un_l, un_r

    def face_state_advection(self, rho, u, v, T, grad_rho, grad_u, grad_v, grad_T):
        rho_vi, rho_vj = rho[self.vi], rho[self.vj]
        u_vi, u_vj = u[self.vi], u[self.vj]
        v_vi, v_vj = v[self.vi], v[self.vj]
        T_vi, T_vj = T[self.vi], T[self.vj]
        # left (in) state
        rho_l = self.xp.maximum(rho_vi + self.xp.sum(grad_rho[self.vi] * self.dfi, axis=1), 1e-10)
        u_l = u_vi + self.xp.sum(grad_u[self.vi] * self.dfi, axis=1)
        v_l = v_vi + self.xp.sum(grad_v[self.vi] * self.dfi, axis=1)
        T_l = self.xp.maximum(T_vi + self.xp.sum(grad_T[self.vi] * self.dfi, axis=1), 1e-10)
        p_l = rho_l * T_l / self.gamma
        c_l = self.xp.sqrt(T_l)
        rhoE_l = p_l / (self.gamma - 1) + 0.5 * rho_l * (u_l ** 2 + v_l ** 2)
        # right (out) state
        rho_r = self.xp.maximum(rho_vj + self.xp.sum(grad_rho[self.vj] * self.dfj, axis=1), 1e-10)
        u_r = u_vj + self.xp.sum(grad_u[self.vj] * self.dfj, axis=1)
        v_r = v_vj + self.xp.sum(grad_v[self.vj] * self.dfj, axis=1)
        T_r = self.xp.maximum(T_vj + self.xp.sum(grad_T[self.vj] * self.dfj, axis=1), 1e-10)
        p_r = rho_r * T_r / self.gamma
        c_r = self.xp.sqrt(T_r)
        rhoE_r = p_r / (self.gamma - 1) + 0.5 * rho_r * (u_r ** 2 + v_r ** 2)
        vars_l = (u_l, v_l, T_l, p_l, c_l)
        vars_r = (u_r, v_r, T_r, p_r, c_r)
        U_l = (rho_l, rho_l * u_l, rho_l * v_l, rhoE_l)
        U_r = (rho_r, rho_r * u_r, rho_r * v_r, rhoE_r)
        return vars_l, vars_r, U_l, U_r

    def _face_state_diffusion(self, u, v, T):
        u_f = self.wi * u[self.vi] + self.wj * u[self.vj]
        v_f = self.wi * v[self.vi] + self.wj * v[self.vj]
        T_f = self.wi * T[self.vi] + self.wj * T[self.vj]
        return u_f, v_f, T_f

    def _gradient_diffusion(self, u, v, T, grad_u, grad_v, grad_T):
        grad_ui, grad_uj = grad_u[self.vi], grad_u[self.vj]
        grad_vi, grad_vj = grad_v[self.vi], grad_v[self.vj]
        grad_Ti, grad_Tj = grad_T[self.vi], grad_T[self.vj]
        grad_u_avg = self.wi[:, None] * grad_ui + self.wj[:, None] * grad_uj
        grad_v_avg = self.wi[:, None] * grad_vi + self.wj[:, None] * grad_vj
        grad_T_avg = self.wi[:, None] * grad_Ti + self.wj[:, None] * grad_Tj
        grad_ufn = ((u[self.vj] - u[self.vi]) / self.edge_length)[:, None] * self.dual_norm
        grad_vfn = ((v[self.vj] - v[self.vi]) / self.edge_length)[:, None] * self.dual_norm
        grad_Tfn = ((T[self.vj] - T[self.vi]) / self.edge_length)[:, None] * self.dual_norm
        grad_uft = grad_u_avg - self.xp.sum(grad_u_avg * self.dual_norm, axis=1, keepdims=True) * self.dual_norm
        grad_vft = grad_v_avg - self.xp.sum(grad_v_avg * self.dual_norm, axis=1, keepdims=True) * self.dual_norm
        grad_Tft = grad_T_avg - self.xp.sum(grad_T_avg * self.dual_norm, axis=1, keepdims=True) * self.dual_norm
        grad_uf = grad_ufn + grad_uft
        grad_vf = grad_vfn + grad_vft
        grad_Tf = grad_Tfn + grad_Tft
        return (grad_uf, grad_vf, grad_Tf)

    def _flux_diffusion(self, u, v, T, grad_u, grad_v, grad_T):
        u_f, v_f, T_f = self._face_state_diffusion(u, v, T)
        grad_uf, grad_vf, grad_Tf = self._gradient_diffusion(u, v, T, grad_u, grad_v, grad_T)
        grad_ux, grad_uy = grad_uf[:, 0], grad_uf[:, 1]
        grad_vx, grad_vy = grad_vf[:, 0], grad_vf[:, 1]
        grad_Tx, grad_Ty = grad_Tf[:, 0], grad_Tf[:, 1]
        mu_f = T_f ** 1.5 * (1 + self.C_inf) / (T_f + self.C_inf)
        k_f = -mu_f / (self.Re_inf * self.Pr * (self.gamma - 1))
        div = grad_ux + grad_vy
        tau_xx = mu_f * (2.0 * grad_ux - (2.0 / 3.0) * div)
        tau_yy = mu_f * (2.0 * grad_vy - (2.0 / 3.0) * div)
        tau_xy = mu_f * (grad_uy + grad_vx)
        indices = self.xp.hstack([self.vi, self.vj])
        Fd1 = self.xp.zeros_like(u)
        Fd2_f = self.dual_length * (tau_xx * self.fnx + tau_xy * self.fny) / self.Re_inf
        Fd3_f = self.dual_length * (tau_xy * self.fnx + tau_yy * self.fny) / self.Re_inf
        Fd2 = self._scatter_aggregate(indices, self.xp.hstack([Fd2_f, -Fd2_f]), self.n_idx, 'add')
        Fd3 = self._scatter_aggregate(indices, self.xp.hstack([Fd3_f, -Fd3_f]), self.n_idx, 'add')
        Fd4_f_stress = self.dual_length * ((tau_xx * u_f + tau_xy * v_f) * self.fnx + (
                tau_xy * u_f + tau_yy * v_f) * self.fny) / self.Re_inf
        Fd4_f_heat = self.dual_length * k_f * (grad_Tx * self.fnx + grad_Ty * self.fny)
        Fd4_stress = self._scatter_aggregate(indices, self.xp.hstack([Fd4_f_stress, -Fd4_f_stress]), self.n_idx, 'add')
        Fd4_heat = self._scatter_aggregate(indices, self.xp.hstack([Fd4_f_heat, -Fd4_f_heat]), self.n_idx, 'add')
        Fd4 = Fd4_stress - Fd4_heat
        Fd = self.xp.column_stack([Fd1, Fd2, Fd3, Fd4])
        return Fd

    def _spect_radius(self, rho, u, v, T):
        rho_f = self.wi * rho[self.vi] + self.wj * rho[self.vj]
        u_f = self.wi * u[self.vi] + self.wj * u[self.vj]
        v_f = self.wi * v[self.vi] + self.wj * v[self.vj]
        T_f = self.wi * T[self.vi] + self.wj * T[self.vj]
        c_f = self.xp.sqrt(T_f)
        mu_f = T_f ** 1.5 * (1.0 + self.C_inf) / (T_f + self.C_inf)
        Lambda_af = (self.xp.abs(u_f * self.fnx + v_f * self.fny) + c_f) * self.dual_length
        Lambda_a = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(Lambda_af, 2), self.n_idx, 'add')
        Lambda_df = self.xp.maximum(4.0 / (3.0 * rho_f), self.gamma / rho_f) * mu_f * self.dual_length ** 2 / self.Pr
        Lambda_d = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(Lambda_df, 2), self.n_idx,
                                           'add') / self.dual_areas
        return Lambda_a, Lambda_d

    def _get_physics(self, rho, rhou, rhov, rhoE):
        u = rhou / rho
        v = rhov / rho
        E = rhoE / rho
        T = self.xp.maximum(self.gamma * (E - 0.5 * (u ** 2 + v ** 2)) * (self.gamma - 1.0), 1e-10)
        return u, v, T

    def _gradient_lsm(self, phi):
        # raw gradient without limitation, Least-Square Gradient
        del_phi = phi[self.lsm_neighbors] - phi[:, None]
        grad_phi = self.xp.einsum('nij,nj->ni', self.lsm_matrices, del_phi)
        return grad_phi

    def _div_boundaries(self, grad_rho, grad_u, grad_v, grad_T):
        if self.T_surf is not None:  # isothermal wall
            raise NotImplementedError
        else:  # adiabatic wall
            grad_u[self.surf_vidx] = self.xp.sum(
                grad_u[self.surf_vidx] * self.surf_norm, axis=1, keepdims=True) * self.surf_norm
            grad_v[self.surf_vidx] = self.xp.sum(
                grad_v[self.surf_vidx] * self.surf_norm, axis=1, keepdims=True) * self.surf_norm
        grad_rho[self.outlet_vidx] = grad_rho[self.outlet_vidx] - self.xp.sum(
            grad_rho[self.outlet_vidx] * self.outlet_norm, axis=1, keepdims=True) * self.outlet_norm
        grad_u[self.outlet_vidx] = grad_u[self.outlet_vidx] - self.xp.sum(
            grad_u[self.outlet_vidx] * self.outlet_norm, axis=1, keepdims=True) * self.outlet_norm
        grad_v[self.outlet_vidx] = grad_v[self.outlet_vidx] - self.xp.sum(
            grad_v[self.outlet_vidx] * self.outlet_norm, axis=1, keepdims=True) * self.outlet_norm
        grad_T[self.outlet_vidx] = grad_T[self.outlet_vidx] - self.xp.sum(
            grad_T[self.outlet_vidx] * self.outlet_norm, axis=1, keepdims=True) * self.outlet_norm
        return grad_rho, grad_u, grad_v, grad_T

    def _fix_boundaries(self, rho, rhou, rhov, rhoE):
        # density and energy truncation
        rho = self.xp.maximum(rho, 1e-10)
        rhoE = self.xp.maximum(rhoE, 1e-10)
        if self.T_surf is not None:  # isothermal wall
            raise NotImplementedError
        else:  # adiabatic wall
            rho[self.surf_vidx] = rho[self.inner_surf_vidx]
            rhou[self.surf_vidx] = 0.0
            rhov[self.surf_vidx] = 0.0
            rho_inner = rho[self.inner_surf_vidx]
            u_inner = rhou[self.inner_surf_vidx] / rho_inner
            v_inner = rhov[self.inner_surf_vidx] / rho_inner
            rhoE[self.surf_vidx] = rhoE[self.inner_surf_vidx] - 0.5 * rho_inner * (u_inner ** 2 + v_inner ** 2)
        rho[self.inlet_vidx] = 1.0
        rhou[self.inlet_vidx] = self.mx_inf
        rhov[self.inlet_vidx] = self.my_inf
        rhoE[self.inlet_vidx] = 1.0 / (self.gamma * (self.gamma - 1)) + self.m_inf ** 2 / 2.0
        rho[self.outlet_vidx] = rho[self.inner_outlet_vidx]
        rhou[self.outlet_vidx] = rhou[self.inner_outlet_vidx]
        rhov[self.outlet_vidx] = rhov[self.inner_outlet_vidx]
        rhoE[self.outlet_vidx] = rhoE[self.inner_outlet_vidx]
        return rho, rhou, rhov, rhoE

    def solve(self, cfl_step=10000, cfl_factor=1.0, cfl_bound=1.0, warm_up=10000):
        rho0, rhou0, rhov0, rhoE0 = self._initialize()
        i = 0
        surf_T_mean = [1.0]
        while i < self.max_iter:
            if i > 0 and i % cfl_step == 0:
                self.cfl = self.xp.minimum(self.cfl * cfl_factor, cfl_bound)
            u0, v0, T0 = self._get_physics(rho0, rhou0, rhov0, rhoE0)
            grad_rho0_raw = self._gradient_lsm(rho0)
            grad_u0_raw = self._gradient_lsm(u0)
            grad_v0_raw = self._gradient_lsm(v0)
            grad_T0_raw = self._gradient_lsm(T0)
            grad_rho0_raw, grad_u0_raw, grad_v0_raw, grad_T0_raw = self._div_boundaries(
                grad_rho0_raw, grad_u0_raw, grad_v0_raw, grad_T0_raw)
            grad_rho0 = self.limiter(rho0, grad_rho0_raw)
            grad_u0 = self.limiter(u0, grad_u0_raw)
            grad_v0 = self.limiter(v0, grad_v0_raw)
            grad_T0 = self.limiter(T0, grad_T0_raw)
            vars_l, vars_r, U_l, U_r = self.face_state_advection(
                rho0, u0, v0, T0, grad_rho0, grad_u0, grad_v0, grad_T0)
            if i <= warm_up:
                Fa = self._rusanov_solver(vars_l, vars_r, U_l, U_r)
            else:
                Fa = self.riemann(vars_l, vars_r, U_l, U_r)
            Fd = self._flux_diffusion(u0, v0, T0, grad_u0_raw, grad_v0_raw, grad_T0_raw)
            Res = (Fa - Fd) / self.dual_areas[:, None]
            Lambda_a, _ = self._spect_radius(rho0, u0, v0, T0)
            dt = self.cfl * self.dual_areas / Lambda_a
            if not self.implicit:
                dU = -dt[:, None] * Res
            else:
                raise NotImplementedError
            rho1 = rho0 + dU[:, 0]
            rhou1 = rhou0 + dU[:, 1]
            rhov1 = rhov0 + dU[:, 2]
            rhoE1 = rhoE0 + dU[:, 3]
            rho1, rhou1, rhov1, rhoE1 = self._fix_boundaries(rho1, rhou1, rhov1, rhoE1)
            u1, v1, T1 = self._get_physics(rho1, rhou1, rhov1, rhoE1)
            p1 = rho1 * T1 / self.gamma
            if self.T_surf is not None:
                raise NotImplementedError
            else:
                surf_T_mean.append(float(T1[self.surf_vidx].mean()))
                if i >= 999:
                    surf_T_mean.pop(0)
                criterion = self.xp.std(self.xp.array(surf_T_mean))
                print(f'Iteration status: iter={i}, criterion={criterion}')
                if len(surf_T_mean) == 1000 and criterion < self.tol:
                    print(f'Converged to solution after {i + 1} iterations!')
                    if self.gpu:
                        rho1, u1, v1, T1, p1 = utils.array2cpu((rho1, u1, v1, T1, p1))
                    return rho1, u1, v1, T1, p1
            i += 1
            rho0, rhou0, rhov0, rhoE0 = rho1, rhou1, rhov1, rhoE1
        print(f'No convergence at max iterations of {self.max_iter}!')
        if self.gpu:
            rho1, u1, v1, T1, p1 = utils.array2cpu((rho1, u1, v1, T1, p1))
        return rho1, u1, v1, T1, p1


class BSWNSFVMSolver(CPGNSFVMSolver, BluntBowShock2D):
    def __init__(self, m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y, L_ref, aoa=None,
                 limiter='venkata', riemann='asumpp', max_iter=5000, tol=1e-8, implicit=False, cfl=0.5, gpu=False):
        CPGNSFVMSolver.__init__(self, limiter, riemann, max_iter, tol, implicit, cfl, gpu)
        BluntBowShock2D.__init__(self, m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y,
                                 L_ref, aoa)
        self._setup_inflow()


class CPGEulerFVMSolver:
    """
        Finite Volume Method for Steady Compressible Euler in Unstructured Mesh
        Cell type: Vertex-centered
        Pseudo time order: 1st (Euler forward/backward difference)
        Spatial order: MUSCL 2nd
        Model type: Calorically Perfect Gas (CPG)
        Conserved Vars: U = (ρ, ρu, ρv, ρE)
        Dimensionalization:
            ρ* = ρ/ρ∞
            u*, v* = u/c∞, v/c∞
            p* = p/(ρ∞·(c∞)^2)
            T* = T/T∞
            E* = E/(c∞)^2
        Gradient limiters:
            Minmod: https://doi.org/10.1016/0021-9991(83)90136-5
            VanLeer: https://doi.org/10.1016/0021-9991(74)90019-9
            Venkatakrishnan: https://doi.org/10.1006/jcph.1995.1084
            Multi-dimensional limiting process (mlp): https://doi.org/10.1016/j.compfluid.2012.04.015
        Riemann dns_solvers:
            Rusanov: https://doi.org/10.1016/0041-5553(62)90062-9
            ASUM+ (asump): https://doi.org/10.1006/jcph.1996.0256
            ASUM+up (asumpp): https://doi.org/10.1016/j.jcp.2005.09.020
            ASUMPW+ (asumpwp): https://doi:10.1006/jcph.2001.6873
            Roe: https://doi.org/10.1006/jcph.1997.5705
                 https://doi.org/10.1007/b79761 (Chapter 11)
        """

    def __init__(self, limiter='venkata', riemann='asumpp', max_iter=5000, tol=1e-8, implicit=False, cfl=1., gpu=False):
        if limiter not in ('minmod', 'vanleer', 'venkata', 'mlp'):
            raise ValueError(f'Unrecognized flux limiter! Please select one from (minmod, vanleer, venkata, mlp)!')
        if riemann not in ('rusanov', 'asump', 'asumpp', 'asumpwp', 'roe'):
            raise ValueError(
                f'Unrecognized riemann solver! Please select one from (rusanov, asump, asumpp, asumpwp, roe)!')
        if not implicit and cfl >= 1.:
            warnings.warn(f'CFL={cfl:.2f} exceeds the theoretical stability limit of 1.0!', RuntimeWarning)
        flux_limiters = {'minmod': self._minmod_limiter,
                         'vanleer': self._vanleer_limiter,
                         'venkata': self._venkata_limiter,
                         'mlp': self._mlp_limiter}
        riemann_solvers = {'rusanov': self._rusanov_solver,
                           'asump': self._asump_solver,
                           'asumpp': self._asumpp_solver,
                           'asumpwp': self._asumpwp_solver,
                           'roe': self._roe_solver}
        self.limiter = flux_limiters[limiter]
        self.riemann = riemann_solvers[riemann]
        self.max_iter = max_iter
        self.tol = tol
        self.implicit = implicit
        self.cfl = cfl  # initial cfl number
        self.gpu = gpu
        self._setup_device()
        self._setup_constants()
        self.n_idx = None
        self.m_inf = None
        self.rho_inf = None
        self.T_inf = None
        self.L = None
        self.aoa = None
        self.pos = None
        self.edge_nodes = None
        self.edge_cells = None
        self.edge_length = None
        self.dual_areas = None
        self.dual_length = None
        self.dual_norm = None
        self.dual_centers = None
        self.surf_length = None
        self.surf_norm = None
        self.inlet_length = None
        self.inlet_norm = None
        self.outlet_length = None
        self.outlet_norm = None
        self.surf_vidx = None
        self.outlet_vidx = None
        self.inlet_vidx = None
        self.is_bound = None
        self.inner_surf_vidx = None
        self.inner_inlet_vidx = None
        self.inner_outlet_vidx = None
        self.surf_eidx = None
        self.inlet_eidx = None
        self.outlet_eidx = None
        self.lsm_neighbors = None
        self.lsm_matrices = None

    def _setup_device(self):
        if self.gpu and cp.cuda.runtime.getDeviceCount() > 0:
            self.xp = cp
            self.coo_matrix = gpu_coo_matrix
            self.csr_matrix = gpu_csr_matrix
            self.diags = gpu_diags
            self.bmat = gpu_bmat
            self.spolve = gpu_spsolve
            self.gmres = gpu_gmres
            self.scatter_add = cupyx.scatter_add
            self.scatter_max = cupyx.scatter_max
            self.scatter_min = cupyx.scatter_min

        else:
            self.xp = np
            self.coo_matrix = coo_matrix
            self.csr_matrix = csr_matrix
            self.diags = diags
            self.bmat = bmat
            self.spolve = spsolve
            self.gmres = gmres
            self.scatter_add = np.add.at
            self.scatter_max = np.maximum.at
            self.scatter_min = np.minimum.at

    def _setup_constants(self):
        self.gamma = 1.4  # specific heat ratio
        self.R_air = 287  # specific gas constant of air [J/(kg·K)]
        self.T0 = 288.15  # Sutherland reference temperature [K]
        self.mu0 = 1.789e-5  # Sutherland reference viscosity [Pa·s]
        self.C0 = 110.4  # Sutherland constant for air [K]
        self.Pr = 0.72  # Prandtl number
        self.c_p = self.R_air * self.gamma / (self.gamma - 1)  # specific heat at constant pressure [J/(kg·K)]

    def _setup_inflow(self):
        self.c_inf = np.sqrt(self.gamma * self.R_air * self.T_inf)  # inflow sound velocity [m/s]
        self.mu_inf = self.mu0 * np.power(self.T_inf / self.T0, 1.5) * (self.T0 + self.C0) / (self.T_inf + self.C0)
        self.Re_inf = self.rho_inf * self.c_inf * self.L / self.mu_inf  # inflow relative Reynolds number
        self.C_inf = self.C0 / self.T_inf

    def _initialize(self):
        if self.gpu:
            self.pos = utils.array2gpu(self.pos)
            self.edge_nodes = utils.array2gpu(self.edge_nodes)
            self.edge_cells = utils.array2gpu(self.edge_cells)
            self.edge_length = utils.array2gpu(self.edge_length)
            self.dual_areas = utils.array2gpu(self.dual_areas)
            self.dual_length = utils.array2gpu(self.dual_length)
            self.dual_norm = utils.array2gpu(self.dual_norm)
            self.dual_centers = utils.array2gpu(self.dual_centers)
            self.surf_length = utils.array2gpu(self.surf_length)
            self.surf_norm = utils.array2gpu(self.surf_norm)
            self.inlet_length = utils.array2gpu(self.inlet_length)
            self.inlet_norm = utils.array2gpu(self.inlet_norm)
            self.outlet_length = utils.array2gpu(self.outlet_length)
            self.outlet_norm = utils.array2gpu(self.outlet_norm)
            self.surf_vidx = utils.array2gpu(self.surf_vidx)
            self.inlet_vidx = utils.array2gpu(self.inlet_vidx)
            self.outlet_vidx = utils.array2gpu(self.outlet_vidx)
            self.inner_surf_vidx = utils.array2gpu(self.inner_surf_vidx)
            self.inner_inlet_vidx = utils.array2gpu(self.inner_inlet_vidx)
            self.inner_outlet_vidx = utils.array2gpu(self.inner_outlet_vidx)
            self.is_bound = utils.array2gpu(self.is_bound)
            self.surf_eidx = utils.array2gpu(self.surf_eidx)
            self.inlet_eidx = utils.array2gpu(self.inlet_eidx)
            self.outlet_eidx = utils.array2gpu(self.outlet_eidx)
            self.lsm_neighbors = utils.array2gpu(self.lsm_neighbors)
            self.lsm_matrices = utils.array2gpu(self.lsm_matrices)
        self.vi, self.vj = self.edge_nodes[:, 0], self.edge_nodes[:, 1]
        self.fnx, self.fny = self.dual_norm[:, 0], self.dual_norm[:, 1]
        self.dfi, self.dfj = self.dual_centers - self.pos[self.vi], self.dual_centers - self.pos[self.vj]
        self.e_ij = (self.pos[self.vj] - self.pos[self.vi]) / self.edge_length[:, None]
        self.li, self.lj = self.xp.linalg.norm(self.dfi, axis=1), self.xp.linalg.norm(self.dfj, axis=1)
        self.wi, self.wj = self.lj / (self.li + self.lj), self.li / (self.li + self.lj)
        if self.aoa is not None:
            self.mx_inf, self.my_inf = self.m_inf * np.cos(self.aoa), self.m_inf * np.sin(self.aoa)
        else:
            self.mx_inf, self.my_inf = self.m_inf, 0.0
        rho0 = self.xp.ones(self.n_idx)
        E0 = 1.0 / (self.gamma * (self.gamma - 1)) + self.m_inf ** 2 / 2.0
        rhou0, rhov0 = rho0 * self.mx_inf, rho0 * self.my_inf
        rhoE0 = rho0 * E0
        return rho0, rhou0, rhov0, rhoE0

    def _scatter_aggregate(self, indices, values, minlength, aggr):
        if aggr not in ('add', 'mean', 'max', 'min'):
            raise ValueError('Unrecognized aggregator type! Please select one from (add, mean, max, min)!')
        shape = (minlength,) if values.ndim == 1 else (minlength, values.shape[-1])
        if aggr == 'add':
            output = self.xp.zeros(shape)
            self.scatter_add(output, indices, values)
            return output
        if aggr == 'mean':
            add = self.xp.zeros(shape)
            count = self.xp.zeros(shape)
            self.scatter_add(add, indices, values)
            self.scatter_add(count, indices, self.xp.ones_like(values))
            output = add / (count + 1e-10)
            return output
        if aggr == 'max':
            output = self.xp.full(shape, -self.xp.inf)
            self.scatter_max(output, indices, values)
            return output
        if aggr == 'min':
            output = self.xp.full(shape, self.xp.inf)
            self.scatter_min(output, indices, values)
            return output

    def _slope_ratio(self, phi, grad_phi):
        del_phi = phi[self.vj] - phi[self.vi]
        del_phi_i = self.xp.sum(grad_phi[self.vi] * (self.dual_centers - self.pos[self.vi]), axis=1)
        del_phi_j = self.xp.sum(grad_phi[self.vj] * (self.pos[self.vj] - self.dual_centers), axis=1)
        theta_i = del_phi / self.xp.where(del_phi_i >= 0, del_phi_i + 1e-10, del_phi_i - 1e-10)
        theta_j = del_phi / self.xp.where(del_phi_j >= 0, del_phi_j + 1e-10, del_phi_j - 1e-10)
        return self.xp.column_stack([theta_i, theta_j])
        # theta = self.xp.minimum(theta_i, theta_j)
        # return theta

    def _minmod_limiter(self, phi, grad_phi):
        theta = self._slope_ratio(phi, grad_phi)
        theta_c = self.xp.clip(theta, 0.0, 1.0)
        # weight = self._scatter_aggregate(self.edge_nodes.ravel(), theta_c.ravel(), self.n_idx, 'min')
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(theta_c, 2).ravel(), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _vanleer_limiter(self, phi, grad_phi):
        theta = self._slope_ratio(phi, grad_phi)
        theta_c = (theta + self.xp.abs(theta)) / (1.0 + self.xp.abs(theta))
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), theta_c.ravel(), self.n_idx, 'min')
        # weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(theta_c, 2).ravel(), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _venkata_limiter(self, phi, grad_phi):
        theta = self._slope_ratio(phi, grad_phi)
        theta_c = self.xp.maximum((theta ** 2 + 2.0 * theta) / (theta ** 2 + theta + 2.0), 0.0)
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), theta_c.ravel(), self.n_idx, 'min')
        # weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(theta_c, 2).ravel(), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _mlp_limiter(self, phi, grad_phi):
        phi_max = self.xp.max(phi[self.edge_nodes], axis=1)
        phi_min = self.xp.min(phi[self.edge_nodes], axis=1)
        neigh_max = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(phi_max, 2), self.n_idx, 'max')
        neigh_min = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(phi_min, 2), self.n_idx, 'min')
        del_max_i, del_max_j = neigh_max[self.vi] - phi[self.vi], neigh_max[self.vj] - phi[self.vj]
        del_min_i, del_min_j = neigh_min[self.vi] - phi[self.vi], neigh_min[self.vj] - phi[self.vj]
        del_phi_i = self.xp.sum(grad_phi[self.vi] * (self.dual_centers - self.pos[self.vi]), axis=1)
        del_phi_j = self.xp.sum(grad_phi[self.vj] * (self.dual_centers - self.pos[self.vj]), axis=1)
        del_phi_i = self.xp.where(del_phi_i >= 0, del_phi_i + 1e-10, del_phi_i - 1e-10)
        del_phi_j = self.xp.where(del_phi_j >= 0, del_phi_j + 1e-10, del_phi_j - 1e-10)
        phi_fi = phi[self.vi] + del_phi_i
        phi_fj = phi[self.vj] + del_phi_j
        theta_i = self.xp.maximum((phi[self.vj] - phi[self.vi]) / del_phi_i, 0.0)
        theta_i = self.xp.where(phi_fi > neigh_max[self.vi], del_max_i / del_phi_i, theta_i)
        theta_i = self.xp.where(phi_fi < neigh_min[self.vi], del_min_i / del_phi_i, theta_i)
        theta_j = self.xp.maximum((phi[self.vi] - phi[self.vj]) / del_phi_j, 0.0)
        theta_j = self.xp.where(phi_fj > neigh_max[self.vj], del_max_j / del_phi_j, theta_j)
        theta_j = self.xp.where(phi_fj < neigh_min[self.vj], del_min_j / del_phi_j, theta_j)
        weight = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.hstack([theta_i, theta_j]), self.n_idx, 'min')
        grad_phi_c = weight[:, None] * grad_phi
        return grad_phi_c

    def _rusanov_solver(self, vars_l, vars_r, U_l, U_r):
        Fa_l, Fa_r, un_l, un_r = self._flux_advection(vars_l, vars_r, U_l, U_r)
        *_, c_l = vars_l
        *_, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        # central difference + dissipation
        lambda_max = self.xp.maximum(self.xp.abs(un_l) + c_l, self.xp.abs(un_r) + c_r)
        Fa1_f = self.dual_length * (0.5 * (Fa_l[0] + Fa_r[0]) - 0.5 * lambda_max * (rho_r - rho_l))  # mass
        Fa2_f = self.dual_length * (0.5 * (Fa_l[1] + Fa_r[1]) - 0.5 * lambda_max * (rhou_r - rhou_l))  # axis-x momentum
        Fa3_f = self.dual_length * (0.5 * (Fa_l[2] + Fa_r[2]) - 0.5 * lambda_max * (rhov_r - rhov_l))  # axis-y momentum
        Fa4_f = self.dual_length * (0.5 * (Fa_l[3] + Fa_r[3]) - 0.5 * lambda_max * (rhoE_r - rhoE_l))  # energy
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _asump_solver(self, vars_l, vars_r, U_l, U_r, alpha=3 / 16, beta=1 / 8):
        u_l, v_l, _, p_l, c_l = vars_l
        u_r, v_r, _, p_r, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        un_r = u_r * self.fnx + v_r * self.fny
        H_l, H_r = (rhoE_l + p_l) / rho_l, (rhoE_r + p_r) / rho_r
        cs_l = self.xp.sqrt(2 * (self.gamma - 1) * H_l / (self.gamma + 1))
        cs_r = self.xp.sqrt(2 * (self.gamma - 1) * H_r / (self.gamma + 1))
        cbar_l = cs_l ** 2 / (self.xp.maximum(self.xp.abs(un_l), cs_l))
        cbar_r = cs_r ** 2 / (self.xp.maximum(self.xp.abs(un_r), cs_r))
        c_f = self.xp.minimum(cbar_l, cbar_r)
        m_l, m_r = un_l / c_f, un_r / c_f
        m_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (m_l + self.xp.abs(m_l)),
                               0.25 * (m_l + 1.0) ** 2 + beta * (m_l ** 2 - 1.0) ** 2)
        m_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (m_r - self.xp.abs(m_r)),
                                -0.25 * (m_r - 1.0) ** 2 - beta * (m_r ** 2 - 1.0) ** 2)
        m_f = m_plus + m_minus
        m_f_plus = 0.5 * (m_f + self.xp.abs(m_f))
        m_f_minus = 0.5 * (m_f - self.xp.abs(m_f))
        p_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (1 + self.xp.sign(m_l)),
                               0.25 * (m_l + 1.0) ** 2 * (2.0 - m_l) + alpha * m_l * (m_l ** 2 - 1) ** 2)
        p_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (1 - self.xp.sign(m_r)),
                                0.25 * (m_r - 1.0) ** 2 * (2.0 + m_r) - alpha * m_r * (m_r ** 2 - 1) ** 2)
        p_f = p_l * p_plus + p_r * p_minus
        Fa1_f = self.dual_length * c_f * (rho_l * m_f_plus + rho_r * m_f_minus)
        Fa2_f = self.dual_length * (c_f * (rhou_l * m_f_plus + rhou_r * m_f_minus) + p_f * self.fnx)
        Fa3_f = self.dual_length * (c_f * (rhov_l * m_f_plus + rhov_r * m_f_minus) + p_f * self.fny)
        Fa4_f = self.dual_length * c_f * ((rhoE_l + p_l) * m_f_plus + (rhoE_r + p_r) * m_f_minus)
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _asumpp_solver(self, vars_l, vars_r, U_l, U_r, k=0.1, kp=0.25, ku=0.1, sigma=0.5, alpha=3 / 16, beta=1 / 8):
        # k = O(1), 0 ≤ kp ≤ 1, 0 ≤ ku ≤ 1, sigma ≤ 1
        u_l, v_l, _, p_l, c_l = vars_l
        u_r, v_r, _, p_r, c_r = vars_r
        rho_l, *_, rhoE_l = U_l
        rho_r, *_, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        un_r = u_r * self.fnx + v_r * self.fny
        H_l, H_r = (rhoE_l + p_l) / rho_l, (rhoE_r + p_r) / rho_r
        # cs_l = self.xp.sqrt(2 * (self.gamma - 1) * H_l / (self.gamma + 1))
        # cs_r = self.xp.sqrt(2 * (self.gamma - 1) * H_r / (self.gamma + 1))
        # cbar_l = cs_l ** 2 / (self.xp.maximum(self.xp.abs(un_l), cs_l))
        # cbar_r = cs_r ** 2 / (self.xp.maximum(self.xp.abs(un_r), cs_r))
        # c_f = self.xp.minimum(cbar_l, cbar_r)
        c_f = 0.5 * (c_l + c_r)
        m_l, m_r = un_l / c_f, un_r / c_f
        m2_bar = 0.5 * (m_l ** 2 + m_r ** 2)
        # m_co = self.xp.maximum(k * self.m_inf, 0.3)
        m_co = k * self.m_inf
        m2_o = self.xp.minimum(1, self.xp.maximum(m2_bar, m_co ** 2))
        m_o = self.xp.sqrt(m2_o)
        fa = m_o * (2 - m_o)
        m1_lp, m1_lm = 0.5 * (m_l + self.xp.abs(m_l)), 0.5 * (m_l - self.xp.abs(m_l))
        m1_rp, m1_rm = 0.5 * (m_r + self.xp.abs(m_r)), 0.5 * (m_r - self.xp.abs(m_r))
        m2_lp, m2_lm = 0.25 * (m_l + 1.0) ** 2, -0.25 * (m_l - 1.0) ** 2
        m2_rp, m2_rm = 0.25 * (m_r + 1.0) ** 2, -0.25 * (m_r - 1.0) ** 2
        m_plus = self.xp.where(self.xp.abs(m_l) > 1.0, m1_lp, m2_lp * (1 - 16 * beta * m2_lm))
        m_minus = self.xp.where(self.xp.abs(m_r) > 1.0, m1_rm, m2_rm * (1 + 16 * beta * m2_rp))
        m_p = kp * self.xp.maximum(1 - sigma * m2_bar, 0.0) * (p_r - p_l) / (fa * 0.5 * (rho_l + rho_r) * c_f ** 2)
        m_f = m_plus + m_minus - m_p
        p_plus = self.xp.where(self.xp.abs(m_l) > 1.0, m1_lp / m_l,
                               m2_lp * (2 - m_l - 16 * alpha * (-4 + 5 * fa ** 2) * m_l * m2_lm))
        p_minus = self.xp.where(self.xp.abs(m_r) > 1.0, m1_rm / m_r,
                                m2_rm * (-2 - m_r + 16 * alpha * (-4 + 5 * fa ** 2) * m_r * m2_rp))
        p_u = ku * p_plus * p_minus * (rho_l + rho_r) * fa * c_f * (u_r - u_l)
        p_f = p_plus * p_l + p_minus * p_r - p_u
        rho_f = self.xp.where(m_f > 0.0, rho_l, rho_r)
        Fa1_f = self.dual_length * c_f * m_f * rho_f
        u_f = self.xp.where(Fa1_f > 0, u_l, u_r)
        v_f = self.xp.where(Fa1_f > 0, v_l, v_r)
        H_f = self.xp.where(Fa1_f > 0, H_l, H_r)
        Fa2_f = Fa1_f * u_f + self.dual_length * p_f * self.fnx
        Fa3_f = Fa1_f * v_f + self.dual_length * p_f * self.fny
        Fa4_f = Fa1_f * H_f
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _asumpwp_solver(self, vars_l, vars_r, U_l, U_r, alpha=3 / 16, beta=1 / 8):
        u_l, v_l, _, p_l, c_l = vars_l
        u_r, v_r, _, p_r, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        un_r = u_r * self.fnx + v_r * self.fny
        H_l, H_r = (rhoE_l + p_l) / rho_l, (rhoE_r + p_r) / rho_r
        # H_norm = 0.5 * (H_l + H_r - 0.5 * (u_l ** 2 + v_l ** 2 + u_r ** 2 + v_r ** 2))
        H_norm = self.xp.maximum(0.5 * (H_l + H_r - 0.5 * (u_l ** 2 + v_l ** 2 + u_r ** 2 + v_r ** 2)), 1e-10)
        c_s = self.xp.sqrt(2 * (self.gamma - 1) * H_norm / (self.gamma + 1))
        c_f = self.xp.where(0.5 * (un_l + un_r) > 0, c_s ** 2 / self.xp.maximum(self.xp.abs(un_l), c_s),
                            c_s ** 2 / self.xp.maximum(self.xp.abs(un_r), c_s))
        m_l, m_r = un_l / c_f, un_r / c_f
        m_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (m_l + self.xp.abs(m_l)),
                               0.25 * (m_l + 1.0) ** 2 + beta * (m_l ** 2 - 1.0) ** 2)
        m_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (m_r - self.xp.abs(m_r)),
                                -0.25 * (m_r - 1.0) ** 2 - beta * (m_r ** 2 - 1.0) ** 2)
        mass_f = m_plus + m_minus

        p_plus = self.xp.where(self.xp.abs(m_l) > 1.0, 0.5 * (1 + self.xp.sign(m_l)),
                               0.25 * (m_l + 1.0) ** 2 * (2.0 - m_l) + alpha * m_l * (m_l ** 2 - 1) ** 2)
        p_minus = self.xp.where(self.xp.abs(m_r) > 1.0, 0.5 * (1 - self.xp.sign(m_r)),
                                0.25 * (m_r - 1.0) ** 2 * (2.0 + m_r) - alpha * m_r * (m_r ** 2 - 1) ** 2)
        p_s = p_plus * p_l + p_minus * p_r
        # asumpw version (asumpw+ needs pressure states of higher-order neighbors)
        f_l = self.xp.where((p_s != 0) & (self.xp.abs(m_l) < 1), (p_l / (p_s + 1e-10) - 1), 0.0)
        f_r = self.xp.where((p_s != 0) & (self.xp.abs(m_r) < 1), (p_r / (p_s + 1e-10) - 1), 0.0)
        w = 1.0 - self.xp.minimum(p_l / p_r, p_r / p_l) ** 3
        mbar_plus = self.xp.where(mass_f >= 0, m_plus + m_minus * ((1 - w) * (1 + f_r) - f_l), m_plus * w * (1 + f_l))
        mbar_minus = self.xp.where(mass_f >= 0, m_minus * w * (1 + f_r), m_minus + m_plus * ((1 - w) * (1 + f_l) - f_r))
        Fa1_f = self.dual_length * c_f * (rho_l * mbar_plus + rho_r * mbar_minus)
        Fa2_f = self.dual_length * (c_f * (rhou_l * mbar_plus + rhou_r * mbar_minus) + p_s * self.fnx)
        Fa3_f = self.dual_length * (c_f * (rhov_l * mbar_plus + rhov_r * mbar_minus) + p_s * self.fny)
        Fa4_f = self.dual_length * c_f * ((rhoE_l + p_l) * mbar_plus + (rhoE_r + p_r) * mbar_minus)
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _roe_solver(self, vars_l, vars_r, U_l, U_r):
        Fa_l, Fa_r, *_ = self._flux_advection(vars_l, vars_r, U_l, U_r)
        u_l, v_l, _, p_l, _ = vars_l
        u_r, v_r, _, p_r, _ = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        wl_rho = self.xp.sqrt(rho_l)
        wl_rhou = wl_rho * u_l
        wl_rhov = wl_rho * v_l
        wl_rhoH = wl_rho * (rhoE_l / rho_l + p_l)
        wr_rho = self.xp.sqrt(rho_r)
        wr_rhou = wr_rho * u_r
        wr_rhov = wr_rho * v_r
        wr_rhoH = wr_rho * (rhoE_r / rho_r + p_r)
        rho_roe = wl_rho + wr_rho
        u_roe = (wl_rhou + wr_rhou) / rho_roe
        v_roe = (wl_rhov + wr_rhov) / rho_roe
        H_roe = (wl_rhoH + wr_rhoH) / rho_roe
        q2_roe = u_roe ** 2 + v_roe ** 2
        T_roe = self.xp.maximum((self.gamma - 1) * (H_roe - 0.5 * q2_roe), 1e-10)
        c_roe = self.xp.sqrt(T_roe)
        lambda1 = self.xp.abs(u_roe - c_roe)
        lambda2 = self.xp.abs(u_roe)
        lambda3 = lambda2
        lambda4 = self.xp.abs(u_roe + c_roe)
        du1 = rho_r - rho_l
        du2 = rhou_r - rhou_l
        du3 = rhov_r - rhov_l
        du4 = rhoE_r - rhoE_l
        du4_bar = du4 - (du3 - v_roe * du1) * v_roe
        alpha3 = du3 - v_roe * du1
        alpha2 = (self.gamma - 1) * (du1 * (H_roe - u_roe ** 2) + u_roe * du2 - du4_bar) / c_roe ** 2
        alpha1 = (du1 * (u_roe + c_roe) - du2 - c_roe * alpha2) / (2 * c_roe)
        alpha4 = du1 - (alpha1 + alpha2)
        rho_dp = lambda1 * alpha1 + lambda2 * alpha2 + lambda4 * alpha4
        rhou_dp = lambda1 * alpha1 * (u_roe - c_roe) + lambda2 * alpha2 * u_roe + lambda4 * alpha4 * (u_roe + c_roe)
        rhov_dp = (lambda1 * alpha1 + lambda2 * alpha2) * v_roe + lambda3 * alpha3 + lambda4 * alpha4 * v_roe
        rhoE_dp = (lambda1 * alpha1 * (H_roe - u_roe * c_roe) + 0.5 * lambda2 * alpha2 * q2_roe
                   + lambda3 * alpha3 * v_roe + lambda4 * alpha4 * (H_roe + u_roe * c_roe))
        Fa1_f = self.dual_length * (0.5 * (Fa_l[0] + Fa_r[0]) - 0.5 * rho_dp)
        Fa2_f = self.dual_length * (0.5 * (Fa_l[1] + Fa_r[1]) - 0.5 * rhou_dp)
        Fa3_f = self.dual_length * (0.5 * (Fa_l[2] + Fa_r[2]) - 0.5 * rhov_dp)
        Fa4_f = self.dual_length * (0.5 * (Fa_l[3] + Fa_r[3]) - 0.5 * rhoE_dp)
        indices = self.xp.hstack([self.vi, self.vj])
        Fa1 = self._scatter_aggregate(indices, np.hstack([Fa1_f, -Fa1_f]), self.n_idx, 'add')
        Fa2 = self._scatter_aggregate(indices, np.hstack([Fa2_f, -Fa2_f]), self.n_idx, 'add')
        Fa3 = self._scatter_aggregate(indices, np.hstack([Fa3_f, -Fa3_f]), self.n_idx, 'add')
        Fa4 = self._scatter_aggregate(indices, np.hstack([Fa4_f, -Fa4_f]), self.n_idx, 'add')
        return self.xp.column_stack([Fa1, Fa2, Fa3, Fa4])

    def _flux_advection(self, vars_l, vars_r, U_l, U_r):
        # raw flux without dissipation
        u_l, v_l, T_l, p_l, c_l = vars_l
        u_r, v_r, T_r, p_r, c_r = vars_r
        rho_l, rhou_l, rhov_l, rhoE_l = U_l
        rho_r, rhou_r, rhov_r, rhoE_r = U_r
        un_l = u_l * self.fnx + v_l * self.fny
        Fa1_l = rho_l * un_l
        Fa2_l = rho_l * un_l * u_l + p_l * self.fnx
        Fa3_l = rho_l * un_l * v_l + p_l * self.fny
        Fa4_l = (rhoE_l + p_l) * un_l
        un_r = u_r * self.fnx + v_r * self.fny
        Fa1_r = rho_r * un_r
        Fa2_r = rho_r * un_r * u_r + p_r * self.fnx
        Fa3_r = rho_r * un_r * v_r + p_r * self.fny
        Fa4_r = (rhoE_r + p_r) * un_r
        Fa_l = (Fa1_l, Fa2_l, Fa3_l, Fa4_l)
        Fa_r = (Fa1_r, Fa2_r, Fa3_r, Fa4_r)
        return Fa_l, Fa_r, un_l, un_r

    def face_state_advection(self, rho, u, v, T, grad_rho, grad_u, grad_v, grad_T):
        rho_vi, rho_vj = rho[self.vi], rho[self.vj]
        u_vi, u_vj = u[self.vi], u[self.vj]
        v_vi, v_vj = v[self.vi], v[self.vj]
        T_vi, T_vj = T[self.vi], T[self.vj]
        # left (in) state
        rho_l = self.xp.maximum(rho_vi + self.xp.sum(grad_rho[self.vi] * self.dfi, axis=1), 1e-10)
        u_l = u_vi + self.xp.sum(grad_u[self.vi] * self.dfi, axis=1)
        v_l = v_vi + self.xp.sum(grad_v[self.vi] * self.dfi, axis=1)
        T_l = self.xp.maximum(T_vi + self.xp.sum(grad_T[self.vi] * self.dfi, axis=1), 1e-10)
        p_l = rho_l * T_l / self.gamma
        c_l = self.xp.sqrt(T_l)
        rhoE_l = p_l / (self.gamma - 1) + 0.5 * rho_l * (u_l ** 2 + v_l ** 2)
        # right (out) state
        rho_r = self.xp.maximum(rho_vj + self.xp.sum(grad_rho[self.vj] * self.dfj, axis=1), 1e-10)
        u_r = u_vj + self.xp.sum(grad_u[self.vj] * self.dfj, axis=1)
        v_r = v_vj + self.xp.sum(grad_v[self.vj] * self.dfj, axis=1)
        T_r = self.xp.maximum(T_vj + self.xp.sum(grad_T[self.vj] * self.dfj, axis=1), 1e-10)
        p_r = rho_r * T_r / self.gamma
        c_r = self.xp.sqrt(T_r)
        rhoE_r = p_r / (self.gamma - 1) + 0.5 * rho_r * (u_r ** 2 + v_r ** 2)
        vars_l = (u_l, v_l, T_l, p_l, c_l)
        vars_r = (u_r, v_r, T_r, p_r, c_r)
        U_l = (rho_l, rho_l * u_l, rho_l * v_l, rhoE_l)
        U_r = (rho_r, rho_r * u_r, rho_r * v_r, rhoE_r)
        return vars_l, vars_r, U_l, U_r

    def _spect_radius(self, u, v, T):
        u_f = self.wi * u[self.vi] + self.wj * u[self.vj]
        v_f = self.wi * v[self.vi] + self.wj * v[self.vj]
        T_f = self.wi * T[self.vi] + self.wj * T[self.vj]
        c_f = self.xp.sqrt(T_f)
        un_outlet = u[self.outlet_vidx] * self.outlet_norm[:, 0] + v[self.outlet_vidx] * self.outlet_norm[:, 1]
        c_outlet = self.xp.sqrt(T[self.outlet_vidx])
        Lambda_f = (self.xp.abs(u_f * self.fnx + v_f * self.fny) + c_f) * self.dual_length
        Lambda = self._scatter_aggregate(self.edge_nodes.ravel(), self.xp.repeat(Lambda_f, 2), self.n_idx, 'add')
        Lambda[self.inlet_vidx] += (self.m_inf + 1.0) * self.inlet_length
        Lambda[self.outlet_vidx] += (self.xp.abs(un_outlet) + c_outlet) * self.outlet_length
        return Lambda

    def _get_physics(self, rho, rhou, rhov, rhoE):
        u = rhou / rho
        v = rhov / rho
        E = rhoE / rho
        T = self.xp.maximum(self.gamma * (E - 0.5 * (u ** 2 + v ** 2)) * (self.gamma - 1.0), 1e-10)
        return u, v, T

    def _gradient_lsm(self, phi):
        # raw gradient without limitation, Least-Square Gradient
        del_phi = phi[self.lsm_neighbors] - phi[:, None]
        grad_phi = self.xp.einsum('nij,nj->ni', self.lsm_matrices, del_phi)
        return grad_phi

    def _fix_boundaries(self, rho, rhou, rhov, rhoE):
        # density and energy truncation
        rho = self.xp.maximum(rho, 1e-10)
        rho[self.surf_vidx] = rho[self.inner_surf_vidx]
        rho_inner = rho[self.inner_surf_vidx]
        u_inner = rhou[self.inner_surf_vidx] / rho_inner
        v_inner = rhov[self.inner_surf_vidx] / rho_inner
        un_inner = u_inner * self.surf_norm[:, 0] + v_inner * self.surf_norm[:, 1]
        u_surf = u_inner - un_inner * self.surf_norm[:, 0]
        v_surf = v_inner - un_inner * self.surf_norm[:, 1]
        rhou[self.surf_vidx] = rho[self.surf_vidx] * u_surf
        rhov[self.surf_vidx] = rho[self.surf_vidx] * v_surf
        rhoE[self.surf_vidx] = rhoE[self.inner_surf_vidx] - 0.5 * rho_inner * un_inner**2
        rho[self.inlet_vidx] = 1.0
        rhou[self.inlet_vidx] = self.mx_inf
        rhov[self.inlet_vidx] = self.my_inf
        rhoE[self.inlet_vidx] = 1.0 / (self.gamma * (self.gamma - 1)) + self.m_inf ** 2 / 2.0
        rho[self.outlet_vidx] = rho[self.inner_outlet_vidx]
        rhou[self.outlet_vidx] = rhou[self.inner_outlet_vidx]
        rhov[self.outlet_vidx] = rhov[self.inner_outlet_vidx]
        rhoE[self.outlet_vidx] = rhoE[self.inner_outlet_vidx]
        return rho, rhou, rhov, rhoE

    def solve(self, cfl_step=10000, cfl_factor=1.0, cfl_bound=1.0, warm_up=10000):
        rho0, rhou0, rhov0, rhoE0 = self._initialize()
        i = 0
        surf_T_mean = [1.0]
        while i < self.max_iter:
            if i > 0 and i % cfl_step == 0:
                self.cfl = self.xp.minimum(self.cfl * cfl_factor, cfl_bound)
            u0, v0, T0 = self._get_physics(rho0, rhou0, rhov0, rhoE0)
            grad_rho0 = self.limiter(rho0, self._gradient_lsm(rho0))
            grad_u0 = self.limiter(u0, self._gradient_lsm(u0))
            grad_v0 = self.limiter(v0, self._gradient_lsm(v0))
            grad_T0 = self.limiter(T0, self._gradient_lsm(T0))
            vars_l, vars_r, U_l, U_r = self.face_state_advection(
                rho0, u0, v0, T0, grad_rho0, grad_u0, grad_v0, grad_T0)
            if i <= warm_up:
                Fa = self._rusanov_solver(vars_l, vars_r, U_l, U_r)
            else:
                Fa = self.riemann(vars_l, vars_r, U_l, U_r)
            Res = Fa / self.dual_areas[:, None]
            Lambda = self._spect_radius(u0, v0, T0)
            dt = self.cfl * self.dual_areas / Lambda
            if not self.implicit:
                dU = -dt[:, None] * Res
            else:
                raise NotImplementedError
            rho1 = rho0 + dU[:, 0]
            rhou1 = rhou0 + dU[:, 1]
            rhov1 = rhov0 + dU[:, 2]
            rhoE1 = rhoE0 + dU[:, 3]
            rho1, rhou1, rhov1, rhoE1 = self._fix_boundaries(rho1, rhou1, rhov1, rhoE1)
            u1, v1, T1 = self._get_physics(rho1, rhou1, rhov1, rhoE1)
            p1 = rho1 * T1 / self.gamma
            surf_T_mean.append(float(T1[self.surf_vidx].mean()))
            if i >= 999:
                surf_T_mean.pop(0)
            criterion = self.xp.std(self.xp.array(surf_T_mean))
            print(f'Iteration status: iter={i}, criterion={criterion}')
            if len(surf_T_mean) == 1000 and criterion < self.tol:
                print(f'Converged to solution after {i + 1} iterations!')
                if self.gpu:
                    rho1, u1, v1, T1, p1 = utils.array2cpu((rho1, u1, v1, T1, p1))
                return rho1, u1, v1, T1, p1
            i += 1
            rho0, rhou0, rhov0, rhoE0 = rho1, rhou1, rhov1, rhoE1
        print(f'No convergence at max iterations of {self.max_iter}!')
        if self.gpu:
            rho1, u1, v1, T1, p1 = utils.array2cpu((rho1, u1, v1, T1, p1))
        return rho1, u1, v1, T1, p1


class BSWEulerFVMSolver(CPGEulerFVMSolver, BluntBowShock2D):
    def __init__(self, m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y, L_ref, aoa=None,
                 limiter='venkata', riemann='asumpp', max_iter=5000, tol=1e-8, implicit=False, cfl=0.5, gpu=False):
        CPGNSFVMSolver.__init__(self, limiter, riemann, max_iter, tol, implicit, cfl, gpu)
        BluntBowShock2D.__init__(self, m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y,
                                 L_ref, aoa)
        self._setup_inflow()


if __name__ == '__main__':
    import utils
    from plot import *
    import time


    # Test SinePoisson**************************************************************************************************
    def test_sine_solver():
        resolution = (20, 20)
        solver = SinePoissonFDMSolver(resolution=resolution, order=2)
        pos = solver.pos
        u_ij = solver.solve(in_idx=False)
        u_idx = solver.solve(in_idx=True)
        u_ref = solver.reference_solution(in_idx=True)
        error = np.abs(u_idx - u_ref)
        tag = (f'({resolution[0]} × {resolution[1]})')
        plot_regular_contour2D(pos, u_ij, title=f'2nd-order FDM {tag}', w=resolution[0], h=resolution[1])
        plot_irregular_contour2D(pos, error, title='error map')
        print(f'l2_norm_error: {utils.l2_norm_error(u_idx, u_ref)}')
        print(f'inf_norm_error: {utils.inf_norm_error(u_idx, u_ref)}')
        print(f'spearman_coef: {utils.spearman_coef(u_idx, u_ref)}')
        print(f'pearson_coef: {utils.pearson_coef(u_idx, u_ref)}')


    # Test PolynomPoisson***********************************************************************************************
    def test_polynom_solver():
        order = 2
        iter_method = 'newton'
        alpha = 1.0
        resolution = (20, 20)
        solver = PolynomPoissonFDMSolver(resolution, order, iter_method=iter_method, tol=1e-8, max_iter=1000)
        u_ref = solver.reference_solution()
        start = time.perf_counter()
        u_fdm = solver.solve(alpha=alpha)
        end = time.perf_counter()
        cost = end - start
        print(f'cost time: {cost:.2f}s!')
        pos = solver.pos
        error = np.abs(u_fdm - u_ref)
        print(f'l2_norm_error: {utils.l2_norm_error(u_fdm, u_ref)}')
        print(f'inf_norm_error: {utils.inf_norm_error(u_fdm, u_ref)}')
        print(f'spearman_coef: {utils.spearman_coef(u_fdm, u_ref)}')
        print(f'pearson_coef: {utils.pearson_coef(u_fdm, u_ref)}')
        plot_irregular_contour2D(pos, u_fdm, title=f'{iter_method} ({order}order)')
        plot_irregular_contour2D(pos, error, title='error map')


    # Test Liouville****************************************************************************************************
    def test_liouville_solver():
        order = 4
        iter_method = 'halley'
        alpha = 0.2
        solver = LiouvilleFDMSolver(nr=20, ntheta=12, order=order, iter_method=iter_method, tol=1e-8, max_iter=1000)
        u_ref = solver.reference_solution()
        start = time.perf_counter()
        u_fdm = solver.solve(alpha=alpha)
        end = time.perf_counter()
        cost = end - start
        print(f'cost time: {cost:.2f}s!')
        cartes_pos = solver.cartes_pos
        error = np.abs(u_fdm - u_ref)
        print(f'l2_norm_error: {utils.l2_norm_error(u_fdm, u_ref)}')
        print(f'inf_norm_error: {utils.inf_norm_error(u_fdm, u_ref)}')
        print(f'spearman_coef: {utils.spearman_coef(u_fdm, u_ref)}')
        print(f'pearson_coef: {utils.pearson_coef(u_fdm, u_ref)}')
        plot_irregular_contour2D(cartes_pos, u_fdm, title=f'{iter_method} ({order}order)')
        plot_irregular_contour2D(cartes_pos, error, title='error map')
        res = np.mean(solver.coef_matrix @ u_ref + solver._right_term(u_ref))
        print(np.round(res, 4))


    # Test Lid-driven Flow**********************************************************************************************
    def test_ldf_solver():
        order = 2
        iter_method = 'newton'
        upwind = False
        alpha = 0.5
        beta = 1.
        gmres_solve = True
        rtol = 1e-10
        init_file = None
        resolution = (150, 150)
        w, h = resolution
        Re = 100
        solver = LDFFDMSolver(resolution, Re, order, iter_method, max_iter=2000, tol=1e-7, gpu=True)
        start = time.perf_counter()
        u, v, p = solver.solve(upwind=upwind, alpha=alpha, beta=beta, gmres_solve=gmres_solve, rtol=rtol,
                               init_file=init_file)
        end = time.perf_counter()
        cost = end - start
        print(f'cost time: {cost:.2f}s!')
        pos = solver.pos
        plot_regular_contour2D(pos, u, title=f'u ({w}×{h}, Re={Re})', w=w, h=h)
        plot_regular_contour2D(pos, v, title=f'v ({w}×{h}, Re={Re})', w=w, h=h)
        plot_regular_contour2D(pos, p, title=f'p ({w}×{h}, Re={Re})', w=w, h=h)


    # Test Backward-step Flow*******************************************************************************************
    def test_bsf_solver():
        order = 2
        iter_method = 'newton'
        upwind = False
        alpha = 0.6
        beta = 1.
        gmres_solve = True
        rtol = 1e-12
        init_file = None
        u_max = 1.
        resolution = (300, 150)
        w, h = resolution
        Re = 100
        solver = BSFFDMSolver(resolution, Re, order, iter_method, max_iter=2000, tol=1e-8, gpu=True)
        start = time.perf_counter()
        u, v, p = solver.solve(upwind=upwind, alpha=alpha, bata=beta, gmres_solve=gmres_solve, rtol=rtol,
                               init_file=init_file, u_max=u_max)
        end = time.perf_counter()
        cost = end - start
        print(f'cost time: {cost:.2f}s!')
        pos = solver.pos
        triangles = solver.triangles
        plot_irregular_contour2D(pos, u, title=f'u (Re={Re})', triangles=triangles, figsize=(8, 4), shrink=0.875)
        plot_irregular_contour2D(pos, v, title=f'v (Re={Re})', triangles=triangles, figsize=(8, 4), shrink=0.875)
        plot_irregular_contour2D(pos, p, title=f'p (Re={Re})', triangles=triangles, figsize=(8, 4), shrink=0.875)


    # Test Bowshock****************************************************************************************************
    def test_bowshock_NS_solver():
        m_inf = 8
        rho_inf = 0.018
        T_inf = 216.65
        T_surf = None
        wall_start = 1e-3
        wall_growth = 1.08
        num_norm = 121
        left_x = -1.8
        top_y = 3.0
        L_ref = 1
        aoa = None
        limiter = 'venkata'
        riemann = 'asumpp'
        max_iter = 100001
        tol = 4.5e-3
        implicit = False
        cfl = 0.3
        gpu = True
        solver = BSWNSFVMSolver(m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y, L_ref,
                                aoa,
                                limiter, riemann, max_iter, tol, implicit, cfl, gpu)
        cfl_step = 10000
        cfl_factor = 1.0
        cfl_bound = 1.0
        warm_up = 10000

        pos = solver.pos
        triangles = solver.triangles
        start = time.perf_counter()
        rho, u, v, T, p = solver.solve(cfl_step=cfl_step, cfl_factor=cfl_factor, cfl_bound=cfl_bound, warm_up=warm_up)
        end = time.perf_counter()
        cost = end - start
        print(f'cost time: {cost / 60:.2f}min!')
        plot_irregular_contour2D(pos, rho, triangles=triangles, title='rho', figsize=(3.6, 12), shrink=0.85)
        plot_irregular_contour2D(pos, u, triangles=triangles, title='u', figsize=(3.6, 12), shrink=0.85)
        plot_irregular_contour2D(pos, v, triangles=triangles, title='v', figsize=(3.6, 12), shrink=0.85)
        plot_irregular_contour2D(pos, T, triangles=triangles, title='T', figsize=(3.6, 12), shrink=0.85)
        plot_irregular_contour2D(pos, p, triangles=triangles, title='p', figsize=(3.6, 12), shrink=0.85)

    def test_bowshock_Euler_solver():
        m_inf = 8
        rho_inf = 0.018
        T_inf = 216.65
        T_surf = None
        wall_start = 1e-3
        wall_growth = 1.08
        num_norm = 121
        left_x = -1.8
        top_y = 3.0
        L_ref = 1
        aoa = None
        limiter = 'venkata'
        riemann = 'rusanov'
        max_iter = 50001
        tol = 7e-3
        implicit = False
        cfl = 0.3
        gpu = True
        solver = BSWEulerFVMSolver(m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x, top_y,
                                   L_ref, aoa, limiter, riemann, max_iter, tol, implicit, cfl, gpu)
        cfl_step = 10000
        cfl_factor = 1.0
        cfl_bound = 1.0
        warm_up = 0

        pos = solver.pos
        triangles = solver.triangles
        start = time.perf_counter()
        rho, u, v, T, p = solver.solve(cfl_step=cfl_step, cfl_factor=cfl_factor, cfl_bound=cfl_bound, warm_up=warm_up)
        end = time.perf_counter()
        cost = end - start
        print(f'cost time: {cost / 60:.2f}min!')
        plot_irregular_contour2D(pos, rho, triangles=triangles, title='rho', figsize=(3.6, 12), shrink=0.665)
        plot_irregular_contour2D(pos, u, triangles=triangles, title='u', figsize=(3.6, 12), shrink=0.665)
        plot_irregular_contour2D(pos, v, triangles=triangles, title='v', figsize=(3.6, 12), shrink=0.665)
        plot_irregular_contour2D(pos, T, triangles=triangles, title='T', figsize=(3.6, 12), shrink=0.65)
        plot_irregular_contour2D(pos, p, triangles=triangles, title='p', figsize=(3.6, 12), shrink=0.65)



    # test_sine_solver()
    # test_polynom_solver()
    # test_liouville_solver()
    # test_ldf_solver()
    # test_bsf_solver()
    # test_bowshock_NS_solver()
    # test_bowshock_Euler_solver()