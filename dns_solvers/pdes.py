import random
import triangle
import utils
import numpy as np
import matplotlib.tri as tri
import networkx as nx
from scipy.spatial import KDTree


class LinearPoisson2D:
    def __init__(self, resolution, domain=(0., 1., 0., 1.)):
        self.resolution = resolution  # grid cells: (width, height)
        self.domain = domain  # (x0, x1, y0, y1)
        self.nx = int(resolution[0] + 1)  # number of points in axis x
        self.ny = int(resolution[1] + 1)  # number of points in axis y
        self.n_idx = self.nx * self.ny
        self.dx = abs(domain[1] - domain[0]) / resolution[0]
        self.dy = abs(domain[3] - domain[2]) / resolution[1]
        self._build_mesh()

    def _build_mesh(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        x0, x1, y0, y1 = self.domain
        lx = np.linspace(x0, x1, self.nx)
        ly = np.linspace(y0, y1, self.ny)
        self.x_ij, self.y_ij = np.meshgrid(lx, ly, indexing='xy')
        self.pos = np.stack([self.x_ij.flatten(), self.y_ij.flatten()], axis=1)  # (x, y), shape=[n_idx, 2]
        edges = []
        for i in range(self.nx):
            for j in range(self.ny):
                src_id = j * self.nx + i
                if i < self.nx - 1:
                    tag_id = src_id + 1
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
                if j < self.ny - 1:
                    tag_id = src_id + self.nx
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
        self.edge_index = np.array(edges).T.astype(np.int64)  # shape=[2, E]

    def _analytical_term(self, x, y):
        raise NotImplementedError('An implementation of _analytical_term method is required!')

    def _right_term(self, x, y):
        raise NotImplementedError('An implementation of _right_term method is required!')

    def reference_solution(self, in_idx=True):
        if in_idx:  # shape=[n_idx, ]
            u_xy = self._analytical_term(self.pos[:, 0], self.pos[:, 1])
            return u_xy
        else:  # shape=[ny, nx]
            u_ij = self._analytical_term(self.x_ij, self.y_ij)
            return u_ij


class NonlinearPoisson2D:
    def __init__(self, resolution, domain=(0., 1., 0., 1.)):
        self.resolution = resolution  # grid cells: (width, height)
        self.domain = domain  # (x0, x1, y0, y1)
        self.nx = int(resolution[0] + 1)  # number of points in axis x
        self.ny = int(resolution[1] + 1)  # number of points in axis y
        self.n_idx = self.nx * self.ny
        self.dx = abs(domain[1] - domain[0]) / resolution[0]
        self.dy = abs(domain[3] - domain[2]) / resolution[1]
        self._build_mesh()

    def _build_mesh(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        x0, x1, y0, y1 = self.domain
        lx = np.linspace(x0, x1, self.nx)
        ly = np.linspace(y0, y1, self.ny)
        self.x_ij, self.y_ij = np.meshgrid(lx, ly, indexing='xy')
        self.pos = np.stack([self.x_ij.flatten(), self.y_ij.flatten()], axis=1)  # (x, y), shape=[n_idx, 2]
        edges = []
        for i in range(self.nx):
            for j in range(self.ny):
                src_id = j * self.nx + i
                if i < self.nx - 1:
                    tag_id = src_id + 1
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
                if j < self.ny - 1:
                    tag_id = src_id + self.nx
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
        self.edge_index = np.array(edges).T.astype(np.int64)  # shape=[2, E]

    def _analytical_term(self, x, y):
        raise NotImplementedError('An implementation of _analytical_term method is required!')

    def _right_term(self, u):
        raise NotImplementedError('An implementation of _right_term method is required!')

    def _right_partial_term_1st(self, u):
        raise NotImplementedError('An implementation of _right_partial_term_1st method is required!')

    def _right_partial_term_2nd(self, u):
        raise NotImplementedError('An implementation of _right_partial_term_2nd method is required!')

    def reference_solution(self, in_idx=True):
        if in_idx:  # shape=[n_idx, ]
            u_xy = self._analytical_term(self.pos[:, 0], self.pos[:, 1])
            return u_xy
        else:  # shape=[ny, nx]
            u_ij = self._analytical_term(self.x_ij, self.y_ij)
            return u_ij


class SinePoisson(LinearPoisson2D):
    def __init__(self, resolution):
        super().__init__(resolution, domain=(0., 1., 0., 1.))
        self._build_bound()

    def _build_bound(self):
        x_idx, y_idx = self.pos[:, 0], self.pos[:, 1]
        x0, x1, y0, y1 = self.domain
        self.is_bound = (x_idx == x0) | (x_idx == x1) | (y_idx == y0) | (y_idx == y1)  # bool, shape=[n_idx, ]
        self.bound_idx = np.arange(self.n_idx)[self.is_bound]  # int, shape=[2*(nx+ny)-4, ]
        self.internal_idx = np.arange(self.n_idx)[~self.is_bound]

    def _analytical_term(self, x, y):
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        return u

    def _right_term(self, x, y):
        f = 2 * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)
        return f


class PolynomPoisson(NonlinearPoisson2D):
    def __init__(self, resolution):
        super().__init__(resolution, domain=(-np.pi / 6, np.pi / 6, -np.pi / 6, np.pi / 6))
        self._build_bound()

    def _build_bound(self):
        x_idx, y_idx = self.pos[:, 0], self.pos[:, 1]
        x0, x1, y0, y1 = self.domain
        self.is_bound = (x_idx == x0) | (x_idx == x1) | (y_idx == y0) | (y_idx == y1)  # bool, shape=[n_idx, ]
        self.bound_idx = np.arange(self.n_idx)[self.is_bound]  # int, shape=[2*(nx+ny)-4, ]
        self.bound_value = self._analytical_term(x_idx[self.is_bound], y_idx[self.is_bound])
        self.internal_idx = np.arange(self.n_idx)[~self.is_bound]

    def _analytical_term(self, x, y):
        u = np.tan(x + y)
        return u

    def _right_term(self, u):
        f = -4 * (u + u ** 3)
        return f

    def _right_partial_term_1st(self, u):
        pfpu = -4 * (1 + 3 * u ** 2)
        return pfpu

    def _right_partial_term_2nd(self, u):
        pf2pu2 = -24 * u
        return pf2pu2


class Liouville2D:
    def __init__(self, nr, ntheta, center=(0., 0.), R=1.):
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.n_idx = nr * ntheta + 1
        self.dr = R / nr
        self.dtheta = 2 * np.pi / ntheta
        self.center = center
        self.R = R  # boundary: x^2 + y^2 = r^2
        self._build_mesh()
        self._build_bound()

    def _build_mesh(self):
        lr = np.linspace(0., self.R, self.nr + 1)[1:]
        ltheta = np.linspace(0., 2 * np.pi, self.ntheta + 1)[:-1]
        r_ij, theta_ij = np.meshgrid(lr, ltheta, indexing='xy')
        pos_ij = np.stack([r_ij.flatten(), theta_ij.flatten()], axis=1)
        self.polar_pos = np.vstack([np.array([0., 0.]), pos_ij])
        self.cartes_pos = self.coord_transform(self.polar_pos[:, 0], self.polar_pos[:, 1])
        edges = []
        for i in range(self.nr):
            for j in range(self.ntheta):
                src_id = 1 + j * self.nr + i
                if i < self.nr - 1:
                    tag_id = 1 + j * self.nr + (i + 1)
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
                if i == 0:
                    edges.append([src_id, 0])
                    edges.append([0, src_id])
                j_id = (j + 1) % self.ntheta
                tag_id = 1 + j_id * self.nr + i
                edges.append([src_id, tag_id])
                edges.append([tag_id, src_id])
        self.edge_index = np.array(edges).T.astype(np.int64)

    def _build_bound(self):
        self.center_idx = [0]
        self.centerlap_idx = list(range(1, self.n_idx, self.nr))  # first inner lap
        r_idx = self.polar_pos[:, 0]
        self.is_bound = r_idx == self.R
        self.bound_idx = np.arange(self.n_idx)[self.is_bound]  # first outer lap
        self.internal_idx = np.setdiff1d(np.arange(self.n_idx)[~self.is_bound], self.center_idx + self.centerlap_idx)

    def _analytical_term(self, r):
        u = np.log(4 / (1 + r ** 2) ** 2)
        return u

    def _right_term(self, u):
        f = -2 * np.exp(u)
        return f

    def _right_partial_term_1st(self, u):
        pfpu = -2 * np.exp(u)
        return pfpu

    def _right_partial_term_2nd(self, u):
        pf2pu2 = -2 * np.exp(u)
        return pf2pu2

    def coord_transform(self, r, theta):
        # Polar coords → Cartesian coords
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        cartes_pos = np.column_stack([x, y])
        return cartes_pos

    def reference_solution(self):
        u = self._analytical_term(self.polar_pos[:, 0])
        return u


class LidDrivenFlow2D:
    def __init__(self, resolution, Re, domain=(0., 1., 0., 1.)):
        self.resolution = resolution
        self.Re = Re  # Reynolds number
        self.domain = domain  # (x0, x1, y0, y1)
        self.nx = int(resolution[0] + 1)  # number of points in axis x
        self.ny = int(resolution[1] + 1)  # number of points in axis y
        self.n_idx = self.nx * self.ny
        self.dx = abs(domain[1] - domain[0]) / resolution[0]
        self.dy = abs(domain[3] - domain[2]) / resolution[1]
        self._build_mesh()
        self._build_bound()

    def _build_mesh(self):
        x0, x1, y0, y1 = self.domain
        lx = np.linspace(x0, x1, self.nx)
        ly = np.linspace(y0, y1, self.ny)
        self.x_ij, self.y_ij = np.meshgrid(lx, ly, indexing='xy')
        self.pos = np.stack([self.x_ij.flatten(), self.y_ij.flatten()], axis=1)  # (x, y), shape=[n_idx, 2]
        edges = []
        for i in range(self.nx):
            for j in range(self.ny):
                src_id = j * self.nx + i
                if i < self.nx - 1:
                    tag_id = src_id + 1
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
                if j < self.ny - 1:
                    tag_id = src_id + self.nx
                    edges.append([src_id, tag_id])
                    edges.append([tag_id, src_id])
        self.edge_index = np.array(edges).T.astype(np.int64)  # shape=[2, E]
        self.mesh_idx = np.arange(self.n_idx).reshape(self.ny, self.nx)

    def _build_bound(self):
        """
        Dirichlet boundary conditions:
            top boundary: u(y=y1) = 4x(1-x); v(y=y1) = 0
            wall (left, right, bottom) boundary: u(x=x0|x=x1|y=y0) = 0; v(x=x0|x=x1|y=y0) = 0
            origin boundary: p(x=x0,y=y0) = 0
        Neumann boundary condition:
            wall boundary: pppn = 0
        """
        x_idx = self.pos[:, 0]
        top_idx = self.mesh_idx[self.ny - 1, :]
        bottom_idx = self.mesh_idx[0, :]
        left_idx = self.mesh_idx[1:-1, 0]
        right_idx = self.mesh_idx[1:-1, self.nx - 1]
        wall_idx = np.hstack([bottom_idx, left_idx, right_idx])
        # origin_idx = self.mesh_idx[0, 0]
        origin_idx = self.mesh_idx[1, 1]  # ensure a full-rank Jacobian, move dirichlet boundary to an interior point
        top_xi = x_idx[top_idx]
        top_xi[1:-1] += 0.5 * self.dx  # staggered offset: x[i, j] = x[x_i+0.5dx, y_j]
        top_u = 4 * top_xi * (1 - top_xi)
        # top_u = np.ones_like(top_idx, dtype=np.float64)
        top_v = np.zeros_like(top_idx, dtype=np.float64)
        wall_uv = np.zeros_like(wall_idx, dtype=np.float64)
        origin_p = np.zeros_like(origin_idx, dtype=np.float64)
        self.bound_idx = np.hstack([top_idx, bottom_idx, left_idx, right_idx])
        self.is_bound = np.isin(np.arange(self.pos.shape[0]), self.bound_idx)
        self.dirichlet_labs = {'u': np.hstack([top_idx, wall_idx]),
                               'v': np.hstack([top_idx, wall_idx]),
                               'p': np.atleast_1d(origin_idx)}
        self.dirichlet_vals = {'u': np.hstack([top_u, wall_uv]),
                               'v': np.hstack([top_v, wall_uv]),
                               'p': np.atleast_1d(origin_p)}
        self.neumann_srcs = {'u': None,
                             'v': None,
                             'p': np.hstack([top_idx, bottom_idx, left_idx, right_idx])}
        self.neumann_tags = {'u': None,
                             'v': None,
                             'p': np.hstack([top_idx - self.nx, bottom_idx + self.nx, left_idx + 1, right_idx - 1])}

    def reference_solution(self, ref_file):
        u_ref = np.load(ref_file)  # shape=[n, 3], seq: u, v, p
        return u_ref


class BackwardStepFlow2D:
    def __init__(self, resolution, Re, domain=(0., 2., 4., 0., 1., 2.)):
        """
        left bottom: (x0, y0)   step corner: (x1, y1)   right top: (x2, y2)
        Domain segmentation:
            rectangle1: A1(x0, y0)          B1(x2, y0)          C1(x2, y1)      D1(x0, y1)
            rectangle2: A2(x1, y1 + dy)     B2(x2, y1 + dy)     C2(x2, y2)      D2(x1, y2)
        """
        if resolution[0] % 2 != 0 or resolution[1] % 2 != 0:
            raise ValueError(f'Resolution must be even to ensure node-alignment with the step edge!')
        if resolution[1] < 4 * 2:
            raise ValueError(f'Insufficient vertical resolution: min 8 required for 4th-order interface processing!')
        self.resolution = resolution
        self.Re = Re  # Reynolds number
        self.domain = domain  # (x0, x1, x2, y0, y1, y2)
        self.nx1 = int(resolution[0] + 1)
        self.ny1 = int(resolution[1] / 2 + 1)
        self.nx2 = int(resolution[0] / 2 + 1)
        self.ny2 = int(resolution[1] / 2)
        self.n_idx1 = self.nx1 * self.ny1
        self.n_idx2 = self.nx2 * self.ny2
        self.n_idx = self.n_idx1 + self.n_idx2
        self.dx = abs(domain[2] - domain[0]) / resolution[0]
        self.dy = abs(domain[5] - domain[3]) / resolution[1]
        self._build_mesh()
        self._build_bound()
        self._build_tris()

    def _build_mesh(self):
        x0, x1, x2, y0, y1, y2 = self.domain
        lx1, lx2 = np.linspace(x0, x2, self.nx1), np.linspace(x1, x2, self.nx2)
        ly1, ly2 = np.linspace(y0, y1, self.ny1), np.linspace(y1 + self.dy, y2, self.ny2)
        self.x1_ij, self.y1_ij = np.meshgrid(lx1, ly1, indexing='xy')
        self.x2_ij, self.y2_ij = np.meshgrid(lx2, ly2, indexing='xy')
        self.pos1 = np.stack([self.x1_ij.flatten(), self.y1_ij.flatten()], axis=1)
        self.pos2 = np.stack([self.x2_ij.flatten(), self.y2_ij.flatten()], axis=1)
        self.pos = np.vstack([self.pos1, self.pos2])
        edges1, edges2 = [], []
        for i1 in range(self.nx1):
            for j1 in range(self.ny1):
                src_id = j1 * self.nx1 + i1
                if i1 < self.nx1 - 1:
                    tag_id = src_id + 1
                    edges1.append([src_id, tag_id])
                    edges1.append([tag_id, src_id])
                if j1 < self.ny1 - 1:
                    tag_id = src_id + self.nx1
                    edges1.append([src_id, tag_id])
                    edges1.append([tag_id, src_id])
        for i2 in range(self.nx2):
            for j2 in range(self.ny2):
                src_id = self.n_idx1 + j2 * self.nx2 + i2
                if i2 < self.nx2 - 1:
                    tag_id = src_id + 1
                    edges2.append([src_id, tag_id])
                    edges2.append([tag_id, src_id])
                if j2 < self.ny2 - 1:
                    tag_id = src_id + self.nx2
                    edges2.append([src_id, tag_id])
                    edges2.append([tag_id, src_id])
        rect_edges = np.vstack([np.array(edges1), np.array(edges2)])
        link_edges = np.stack(
            [np.arange(self.n_idx1 - self.nx2, self.n_idx1), np.arange(self.n_idx1, self.n_idx1 + self.nx2)], axis=1)
        self.edge_index = np.vstack([rect_edges, link_edges]).T.astype(np.int64)
        self.mesh_idx1 = np.arange(self.n_idx1).reshape(self.ny1, self.nx1)
        self.mesh_idx2 = np.arange(self.n_idx1, self.n_idx).reshape(self.ny2, self.nx2)

    def _build_bound(self):
        """
        Dirichlet boundary conditions:
            inlet: u(x=x0) = 4y(1-y); v(x=x0) = 0
            wall: u(y=y0|y=y2|x∈(x0, x1)|y∈(y1, y2)) = 0, v(y=y0|y=y2|x∈(x0, x1)|y∈(y1, y2))=0
            outlet: p(x=x2) = 0
        Neumann boundary condition:
            inlet: pppx = 0
            wall: pppn = 0
            outlet: pupx = 0; pvpx = 0
        """
        y1_idx = self.pos1[:, 1]
        inlet_idx = self.mesh_idx1[:, 0]
        outlet_idx1 = self.mesh_idx1[:, -1]
        outlet_idx2 = self.mesh_idx2[:, -1]
        outlet_idx = np.hstack([outlet_idx1, outlet_idx2])
        xwall_idx1 = self.mesh_idx1[0, 1:-1]
        xwall_idx2 = self.mesh_idx1[-1, 1: self.nx1 - self.nx2 + 1]
        xwall_idx3 = self.mesh_idx2[-1, :-1]
        xwall_idx = np.hstack([xwall_idx1, xwall_idx2, xwall_idx3])
        ywall_idx = self.mesh_idx2[:-1, 0]
        wall_idx = np.hstack([xwall_idx, ywall_idx])
        inlet_yi = y1_idx[inlet_idx]
        inlet_u = 4 * inlet_yi * (1 - inlet_yi)
        inlet_v = np.zeros_like(inlet_idx, dtype=np.float64)
        outlet_p = np.zeros_like(outlet_idx, dtype=np.float64)
        wall_uv = np.zeros_like(wall_idx, dtype=np.float64)
        self.dirichlet_labs = {'u': np.hstack([inlet_idx, wall_idx]),
                               'v': np.hstack([inlet_idx, wall_idx]),
                               'p': outlet_idx}
        self.dirichlet_vals = {'u': np.hstack([inlet_u, wall_uv]),
                               'v': np.hstack([inlet_v, wall_uv]),
                               'p': outlet_p}
        self.neumann_srcs = {'u': outlet_idx,
                             'v': outlet_idx,
                             'p': np.hstack([inlet_idx, wall_idx])}
        self.neumann_tags = {'u': outlet_idx - 1,
                             'v': outlet_idx - 1,
                             'p': np.hstack(
                                 [inlet_idx + 1, xwall_idx1 + self.nx1, xwall_idx2 - self.nx1, xwall_idx3 - self.nx2,
                                  ywall_idx + 1])}
        self.bound_idx = np.hstack([inlet_idx, wall_idx, outlet_idx])
        # required for neural network training
        self.is_bound = np.isin(np.arange(self.pos.shape[0]), self.bound_idx)
        self.inlet_idx = inlet_idx
        self.wall_idx = wall_idx
        self.outlet_idx = outlet_idx

    def _build_tris(self):
        # saved only for contour plot
        x, y = self.pos[:, 0], self.pos[:, 1]
        tris = tri.Triangulation(x, y)
        tri_indices = tris.triangles
        vertex_c1 = self.mesh_idx1[0, 0]  # left bottom corner (x0, y0)
        vertex_c2 = self.mesh_idx1[0, -1]  # right bottom corner (x2, y0)
        vertex_c3 = self.mesh_idx1[-1, 0]  # left top corner (x0, y1)
        vertex_c4 = self.mesh_idx2[-1, 0]  # medium top corner (x1, y2)
        vertex_c5 = self.mesh_idx2[-1, -1]  # right top corner (x2, y2)
        vertex_mask = np.isin(tri_indices, self.bound_idx)
        invalid_tris = np.all(vertex_mask, axis=1)
        tris.set_mask(invalid_tris)
        triangles = tris.triangles[~invalid_tris]  # remove invalid triangles in concave domain
        triangles = np.vstack([triangles,
                               np.array([[vertex_c1, vertex_c1 + 1, vertex_c1 + self.nx1],
                                         [vertex_c2, vertex_c2 - 1, vertex_c2 + self.nx1],
                                         [vertex_c3, vertex_c3 + 1, vertex_c3 - self.nx1],
                                         [vertex_c4, vertex_c4 + 1, vertex_c4 - self.nx2],
                                         [vertex_c5, vertex_c5 - 1, vertex_c5 - self.nx2]])])
        self.triangles = triangles

    def reference_solution(self, ref_file):
        u_ref = np.load(ref_file)  # shape=[n, 3], seq: u, v, p
        return u_ref


class BluntBowShock2D:
    def __init__(self, m_inf, rho_inf, T_inf, T_surf, wall_start=1e-6, wall_growth=1.1, num_norm=180, left_x=-3.0,
                 top_y=6.0, L_ref=1.0, aoa=None, center=(0., 0.), R=1.0, ):
        self.m_inf = m_inf  # inflow mach number
        self.rho_inf = rho_inf  # inflow density  [kg/m^3]
        self.T_inf = T_inf  # inflow temperature  [K]
        self.T_surf = T_surf  # specify wall temperature if isothermal wall, set to None for adiabatic wall
        self.L = L_ref  # reference length
        self.wall_start = wall_start  # initial wall-normal spacing (only for x-axis direction, directional nonuniform)
        self.wall_growth = wall_growth  # grid stretching ratio (only for x-axis direction, adaptively scaling)
        self.num_norm = num_norm  # number of discrete points on normal direction (cylinder wall surface)
        self.aoa = aoa  # rad (-pi/2, pi/2), if None, aoa=0
        self.center = center
        self.R = R
        self.num_tang = self._tangential_number(-R - left_x)  # number of discrete points on tangential direction
        self._build_mesh(left_x, top_y)
        self._build_LSM_topology()

    def _tangential_number(self, l0):
        k = np.log(l0 * (self.wall_growth - 1.0) / self.wall_start + 1.0) / np.log(self.wall_growth)
        return int(np.floor(k) + 1)

    def _get_outerbound(self, inner_pos, radius_norms, l, R2):
        # get intersection points with outer circle
        inner_x, inner_y = inner_pos[:, 0], inner_pos[:, 1]
        inner_nx, inner_ny = radius_norms[:, 0], radius_norms[:, 1]
        A = inner_nx ** 2 + inner_ny ** 2
        B = 2.0 * (inner_x - l) * inner_nx + 2.0 * inner_y * inner_ny
        C = (inner_nx - l) ** 2 + inner_ny ** 2.0 - R2 ** 2.0
        discriminant = B ** 2.0 - 4.0 * A * C
        t = (-B + np.sqrt(discriminant)) / (2.0 * A)
        outer_x = inner_x + t * inner_nx
        outer_y = inner_y + t * inner_ny
        return np.column_stack([outer_x, outer_y])

    def _get_pos(self, inner_pos, outer_pos, radius_norms):
        dh = outer_pos - inner_pos
        tang_dist = np.linalg.norm(dh, axis=-1)
        h0 = self.wall_start * tang_dist / tang_dist.min()
        residual = tang_dist - h0 * (self.wall_growth ** (self.num_tang - 1.0) - 1.0) / (self.wall_growth - 1.0)
        tang_dl = h0[:, None] * np.power(self.wall_growth, np.arange(self.num_tang - 1.0))[None, :]
        tail_num = np.minimum(10, self.num_tang - 1)
        tang_dl[:, -tail_num:] += residual[:, None] / tail_num
        tang_dl = np.column_stack([np.zeros(self.num_norm), tang_dl])
        radius_l = np.cumsum(tang_dl, axis=1) + self.R
        pos = np.array(self.center)[None, None, :] + radius_l[:, :, None] * radius_norms[:, None, :]
        # point ordering: radial normal (dim0) first, followed by tangential rays nodes (dim1)
        pos = np.transpose(pos, (1, 0, 2))
        return pos.reshape(-1, pos.shape[-1])

    def _rect_cells(self):
        # i: tangential cell base index [0, num_norm-2], shape (num_norm-1, 1)
        # j: radial cell base index [0, num_tang-2], shape (1, num_tang-1)
        i = np.arange(self.num_tang - 1)[:, None]
        j = np.arange(self.num_norm - 1)[None, :]
        p0 = i * self.num_norm + j
        p1 = (i + 1) * self.num_norm + j
        p2 = (i + 1) * self.num_norm + (j + 1)
        p3 = i * self.num_norm + (j + 1)
        cells = np.stack([p0, p1, p2, p3], axis=-1).reshape(-1, 4)
        return cells

    def _kdt_search(self, tree_pos, qurey_pos):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
        tree = KDTree(tree_pos)
        sdf, _ = tree.query(qurey_pos)
        return sdf

    def _binary_search2D(self, tree_edges, query_edges):
        # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        item_size = tree_edges.dtype.itemsize * 2
        # one-shot search: 2D array → 1D binaries
        tree_bit = tree_edges.view(np.dtype((np.void, item_size))).reshape(-1)
        query_bit = query_edges.view(np.dtype((np.void, item_size))).reshape(-1)
        sorter = np.argsort(tree_bit)
        idx = np.searchsorted(tree_bit, query_bit, sorter=sorter)
        query_res = np.full_like(query_bit, -1, dtype=np.int64)
        mask = idx < len(tree_bit)
        if np.any(mask):
            matched_mask = tree_bit[sorter[idx[mask]]] == query_bit[mask]
            actual_mask = np.zeros_like(mask, dtype=bool)
            actual_mask[mask] = matched_mask
            query_res[actual_mask] = sorter[idx[actual_mask]]
        return query_res

    def _build_mesh(self, left_x, top_y):
        """
        Computational domain: an outer circle passing through left_bottom and right_top with its center on the x-axis,
        forming the computational domain together with x-axis, y-axis and the inner circle
            outer circle: (x-l)^2 + y^2 = R2^2 (inlet)
            inner circle: x^2 + y^2 = R^2 (wall surface)
            right: y = 0 (outelt)
        Topological List:
            pos (nodes/vertices): float, 2D position coords, shape=[Nv, 2]
            sdf: float, signed distance field, shape=[Nv, ]
            cells: int, flat connectivity array (counter-clockwise) of all cells, shape=[N1*3 + N2*4 + ..., ]
            edge_nodes: int, node indices sharing each edge, shape=[Ne, 2]
            edge_cells: int, cell indices sharing each edge, [cell_in, cell_out], shape=[Ne, 2]
            cell_offset: int, starting index of each cell in cells, shape=[Nc, ]
            cell_topology: int, number of faces/nodes per cell, shape=[Nc, ]
            Cell-Centered FVM:
                edge_length: float, length of each edge (face), shape=[Ne, ]
                edge_norm: float, normal vector of each edge (face), shape=[Ne, 2]
                cell_areas: float, area of each control cell, shape=[Nc, ]
            Vertex-Centered FVM (dual cells connecting centroids of cells):
                dual_length: float, length of each vertex-centered dual face, shape=[Ne, 2]
                dual_norm: float, normal vector of each vertex-centered dual face
                dual_centers: float, position coords of each dual face, shape=[Ne, 2]
                dual_areas: float, area of each vertex-centered dual cell, shape=[Nv, ]
        """
        l = (left_x ** 2 - top_y ** 2) / (2 * left_x)
        R2 = l - left_x
        ox, oy = self.center
        # inner_rad = np.linspace(0.5 * np.pi, np.pi, self.num_norm)
        inner_rad = np.linspace(0.5 * np.pi, 1.5 * np.pi, self.num_norm)
        surf_pos = np.column_stack([self.R * np.cos(inner_rad) + ox, self.R * np.sin(inner_rad) + oy])
        radius_norms = (surf_pos - np.array(self.center)[None, :]) / self.R
        inlet_pos = self._get_outerbound(surf_pos, radius_norms, l, R2)
        self.pos = self._get_pos(surf_pos, inlet_pos, radius_norms)  # [num_norm * num_tang, 2]
        self.n_idx = self.pos.shape[0]
        mesh_idx = np.arange(self.num_norm * self.num_tang).reshape(self.num_tang, self.num_norm)
        self.surf_vidx = mesh_idx[0, :]
        # self.inner_surf_vidx = mesh_idx[1, :]
        self.inner_surf_vidx = np.hstack([mesh_idx[1, 1], mesh_idx[1, 1:-1], mesh_idx[1, -2]])
        self.inlet_vidx = mesh_idx[-1, :]
        # self.inner_inlet_vidx = mesh_idx[-2, :]
        self.inner_inlet_vidx = np.hstack([mesh_idx[-2, 1], mesh_idx[-2, 1:-1], mesh_idx[-2, -2]])
        self.outlet_vidx = np.hstack([mesh_idx[1:-1, 0], mesh_idx[1:-1, -1]])
        self.inner_outlet_vidx = np.hstack([mesh_idx[1:-1, 1], mesh_idx[1:-1, -2]])
        self.sdf = self._kdt_search(tree_pos=self.pos[self.surf_vidx], qurey_pos=self.pos)
        cells = self._rect_cells()
        self.cells = cells.ravel()
        self.offset = np.arange(cells.shape[0]) * 4
        self.cell_topology = np.ones(cells.shape[0]) * 4
        self.triangles = np.vstack([cells[:, :3], cells[:, [2, 3, 0]]])  # only for contour plot
        edges = np.vstack([cells[:, :2], cells[:, 1:3], cells[:, 2:4], cells[:, [3, 0]]])
        self.edge_nodes = np.unique(np.sort(edges, axis=1), axis=0)  # directed, fixed to (low_idx → higher_idx)
        p0, p1 = self.pos[self.edge_nodes[:, 0]], self.pos[self.edge_nodes[:, 1]]
        self.edge_length = np.linalg.norm(np.abs(p1 - p0), axis=1)  # [ne, ]
        # self.edge_norm = np.column_stack([(p0 - p1)[:, 1], (p1 - p0)[:, 0]]) / self.edge_length[:, None]
        self.edge_norm = np.column_stack([(p1 - p0)[:, 1], (p0 - p1)[:, 0]]) / self.edge_length[:, None]
        surf_edges = np.sort(np.column_stack([self.surf_vidx[:-1], self.surf_vidx[1:]]), axis=1)
        inlet_edges = np.sort(np.column_stack([self.inlet_vidx[:-1], self.inlet_vidx[1:]]), axis=1)
        outlet_edges = np.sort(np.column_stack([np.hstack([mesh_idx[0:-1, 0], mesh_idx[0:-1, -1]]),
                                                np.hstack([mesh_idx[1:, 0], mesh_idx[1:, -1]])]), axis=1)
        self.surf_eidx = self._binary_search2D(self.edge_nodes, surf_edges)
        self.inlet_eidx = self._binary_search2D(self.edge_nodes, inlet_edges)
        self.outlet_eidx = self._binary_search2D(self.edge_nodes, outlet_edges)
        edge2cell = np.tile(np.arange(cells.shape[0]), 4)
        res_in = self._binary_search2D(edges, self.edge_nodes)
        res_out = self._binary_search2D(edges, np.column_stack([self.edge_nodes[:, 1], self.edge_nodes[:, 0]]))
        cells_in, cells_out = edge2cell[res_in], edge2cell[res_out]
        bound_in, bound_out = res_in == -1, res_out == -1
        surf_in = bound_in & np.isin(np.arange(self.edge_nodes.shape[0]), self.surf_eidx)
        surf_out = bound_out & np.isin(np.arange(self.edge_nodes.shape[0]), self.surf_eidx)
        inlet_in = bound_in & np.isin(np.arange(self.edge_nodes.shape[0]), self.inlet_eidx)
        inlet_out = bound_out & np.isin(np.arange(self.edge_nodes.shape[0]), self.inlet_eidx)
        outlet_in = bound_in & np.isin(np.arange(self.edge_nodes.shape[0]), self.outlet_eidx)
        outlet_out = bound_out & np.isin(np.arange(self.edge_nodes.shape[0]), self.outlet_eidx)
        cells_in[surf_in], cells_out[surf_out] = -1, -1
        cells_in[inlet_in], cells_out[inlet_out] = -2, -2
        cells_in[outlet_in], cells_out[outlet_out] = -3, -3
        self.edge_cells = np.column_stack([cells_in, cells_out])
        cell_x, cell_y = self.pos[:, 0][cells], self.pos[:, 1][cells]
        cell_xr, cell_yr = np.roll(cell_x, shift=-1, axis=1), np.roll(cell_y, shift=-1, axis=1)
        # shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
        self.cell_areas = np.sum(cell_x * cell_yr - cell_xr * cell_y, axis=1) / 2.0
        cell_cx = np.sum((cell_x + cell_xr) * (cell_x * cell_yr - cell_xr * cell_y), axis=1) / (6.0 * self.cell_areas)
        cell_cy = np.sum((cell_y + cell_yr) * (cell_x * cell_yr - cell_xr * cell_y), axis=1) / (6.0 * self.cell_areas)
        self.cell_centroids = np.column_stack([cell_cx, cell_cy])
        # vertex-centered dual cells
        contrib_area = self.cell_areas / self.cell_topology
        node2cell = np.repeat(np.arange(cells.shape[0]), 4)
        self.dual_areas = np.bincount(self.cells, weights=contrib_area[node2cell], minlength=self.n_idx)
        edge_centers = 0.5 * (self.pos[self.edge_nodes[:, 0]] + self.pos[self.edge_nodes[:, 1]])
        f0, f1 = np.zeros((self.edge_nodes.shape[0], 2)), np.zeros((self.edge_nodes.shape[0], 2))
        f0[bound_in], f1[bound_out] = edge_centers[bound_in], edge_centers[bound_out]
        f0[~bound_in] = self.cell_centroids[self.edge_cells[~bound_in, 0]]
        f1[~bound_out] = self.cell_centroids[self.edge_cells[~bound_out, 1]]
        self.dual_length = np.linalg.norm(np.abs(f1 - f0), axis=1)
        self.dual_centers = 0.5 * (f1 + f0)
        self.dual_norm = np.column_stack([(f0 - f1)[:, 1], (f1 - f0)[:, 0]]) / self.dual_length[:, None]
        # supplementary dual face for boundary dual cells trunked by boundary edges
        surf_points = np.vstack([(self.pos[mesh_idx[0, 0]] + self.pos[mesh_idx[1, 0]]) / 2.0,
                                 edge_centers[self.surf_eidx],
                                 (self.pos[mesh_idx[0, -1]] + self.pos[mesh_idx[1, -1]]) / 2.0])
        surf_f0, surf_f1 = surf_points[:-1], surf_points[1:]
        self.surf_length = np.linalg.norm(np.abs(surf_f1 - surf_f0), axis=1)
        # self.surf_norm = (np.column_stack([(surf_f0 - surf_f1)[:, 1], (surf_f1 - surf_f0)[:, 0]])
        #                   / self.surf_length[:, None])
        self.surf_norm = self.pos[self.surf_vidx]
        inlet_points = np.vstack([(self.pos[mesh_idx[-2, 0]] + self.pos[mesh_idx[-1, 0]]) / 2.0,
                                  edge_centers[self.inlet_eidx],
                                  (self.pos[mesh_idx[-2, -1]] + self.pos[mesh_idx[-1, -1]]) / 2.0])
        inlet_f0, inlet_f1 = inlet_points[:-1], inlet_points[1:]
        self.inlet_length = np.linalg.norm(np.abs(inlet_f1 - inlet_f0), axis=1)
        self.inlet_norm = (np.column_stack([(inlet_f1 - inlet_f0)[:, 1], (inlet_f0 - inlet_f1)[:, 0]])
                           / self.inlet_length[:, None])
        outlet_f0 = np.vstack([(self.pos[mesh_idx[0:-1, 0]] + self.pos[mesh_idx[1:, 0]])[:-1] / 2.0,
                               (self.pos[mesh_idx[0:-1, -1]] + self.pos[mesh_idx[1:, -1]])[:-1] / 2.0])
        outlet_f1 = np.vstack([(self.pos[mesh_idx[0:-1, 0]] + self.pos[mesh_idx[1:, 0]])[1:] / 2.0,
                               (self.pos[mesh_idx[0:-1, -1]] + self.pos[mesh_idx[1:, -1]])[1:] / 2.0])
        self.outlet_length = np.linalg.norm(np.abs(outlet_f1 - outlet_f0), axis=1)
        self.outlet_norm = np.tile([1.0, 0.0], ((self.num_tang - 2) * 2, 1))
        self.bound_vidx = np.hstack([self.surf_vidx, self.inlet_vidx, self.outlet_vidx])
        self.is_bound = np.isin(np.arange(self.n_idx), self.bound_vidx)


    def _build_LSM_topology(self, k=8):
        # k must be less than or equal to the total count of 1st and 2nd order neighbors
        # if insufficient, please enable 3rd-order neighbor search to maintain matrix stability
        cells = self.cells.reshape(-1, 4)
        edges_pairs = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 3]], cells[:, [3, 0]]])
        diag_pairs = np.vstack([cells[:, [0, 2]], cells[:, [1, 3]]])
        edges = np.unique(np.sort(np.vstack([edges_pairs, diag_pairs])), axis=0)  # diagonal neighbors
        G = nx.Graph()
        G.add_edges_from(edges)
        neighbors = []
        for v in range(self.pos.shape[0]):
            neigh = []
            neigh1 = list(G.neighbors(v))  # first-order neighbors
            if len(neigh1) >= k:
                neigh.extend(neigh1[:k])
            else:
                neigh.extend(neigh1)
                neigh2 = []  # second-order neighbors
                for n in neigh1:
                    neigh2.extend(list(G.neighbors(n)))
                neigh2 = set(filter(lambda x: x != v and x not in neigh1, neigh2))
                neigh.extend(random.sample(neigh2, k - len(neigh1)))
                # print(f'id: {v}, neigh: {neigh}')
            neighbors.append(neigh)
        self.lsm_neighbors = np.array(neighbors, dtype=np.int64)
        dh = self.pos[self.lsm_neighbors] - self.pos[:, None, :]
        dist = np.linalg.norm(dh, axis=-1)
        weights = 1 / (dist + 1e-6) / np.sum(1 / (dist + 1e-6), axis=1)[:, None]
        W_A = weights[:, :, None] * dh
        ATA = np.einsum('nki,nkj->nij', dh, W_A)
        ATA_inv = np.linalg.inv(ATA)
        AT_W = W_A.transpose(0, 2, 1)
        self.lsm_matrices = np.einsum('nij,njk->nik', ATA_inv, AT_W)

    def reference_solution(self, ref_file=None):
        u_ref = np.load(ref_file)  # shape=[n, 5], seq: rho, u, v, T, p
        return u_ref


if __name__ == '__main__':
    from plot import *


    # Test SinePoisson**************************************************************************************************
    def test_sine():
        resolution = (50, 50)
        sine = SinePoisson(resolution=resolution)
        pos, edge_index = sine.pos, sine.edge_index
        u = sine.reference_solution(in_idx=True)
        u_ij = sine.reference_solution(in_idx=False)
        plot_mesh(pos, edge_index)
        w, h = resolution
        plot_regular_contour2D(pos=(sine.x_ij, sine.y_ij), u=u_ij, title='Analytical solution')
        plot_regular_contour2D(pos, u, w=w, h=h, title='Analytical solution')


    # Test PolynomPoisson***********************************************************************************************
    def test_polynom():
        resolution = (50, 50)
        polynom = PolynomPoisson(resolution=resolution)
        pos, edge_index = polynom.pos, polynom.edge_index
        u = polynom.reference_solution(in_idx=True)
        u_ij = polynom.reference_solution(in_idx=False)
        plot_mesh(pos, edge_index)
        w, h = resolution
        pfpu = polynom._right_partial_term_1st(u)
        plot_regular_contour2D(pos=(polynom.x_ij, polynom.y_ij), u=u_ij, title='Analytical solution')
        plot_regular_contour2D(pos, u, w=w, h=h, title='Analytical solution')
        plot_regular_contour2D(pos, pfpu, w=w, h=h, title='-(u_xx + u_yy)')


    # Test Liouville****************************************************************************************************
    def test_liouville():
        liuville = Liouville2D(nr=50, ntheta=24)
        cartes_pos = liuville.cartes_pos
        edge_index = liuville.edge_index
        u = liuville.reference_solution()
        pfpu = liuville._right_partial_term_1st(u)
        plot_mesh(cartes_pos, edge_index)
        plot_irregular_contour2D(cartes_pos, u)
        plot_irregular_contour2D(cartes_pos, pfpu)


    # Test Lid-driven Flow**********************************************************************************************
    def test_ldf():
        ldf = LidDrivenFlow2D(resolution=(50, 50), Re=100)
        pos = ldf.pos
        edge_index = ldf.edge_index
        u_random = np.random.randn(pos.shape[0])
        plot_mesh(pos, edge_index)
        plot_irregular_contour2D(pos, u_random)


    # Test Backward-step Flow*******************************************************************************************
    def test_bsf():
        bsf = BackwardStepFlow2D(resolution=(100, 50), Re=100)
        pos = bsf.pos
        edge_index = bsf.edge_index
        u_random = np.random.randn(pos.shape[0])
        triangles = bsf.triangles
        plot_mesh(pos, edge_index, figsize=(8, 4))
        plot_irregular_contour2D(pos, u_random, triangles=triangles, figsize=(8, 4), shrink=0.85)


    # Test CylinderHypersonicFlow **************************************************************************************
    def test_bowshock():
        m_inf = 8
        rho_inf = 0.018
        T_inf = 216.65
        T_surf = None
        wall_start = 1e-3
        wall_growth = 1.08
        num_norm = 121
        bowshock = BluntBowShock2D(m_inf, rho_inf, T_inf, T_surf, wall_start, wall_growth, num_norm, left_x=-1.8,
                                   top_y=3.0)
        pos = bowshock.pos
        edge_nodes = bowshock.edge_nodes
        print(bowshock.num_norm, bowshock.num_tang)
        print(f'num_nodes: {pos.shape[0]}, num_edges: {edge_nodes.shape[0]}')
        plot_mesh(pos, edge_nodes.T, figsize=(2, 6))
        u_temp = np.random.randn(pos.shape[0])
        plot_irregular_contour2D(pos, u_temp, triangles=bowshock.triangles, figsize=(2, 6), shrink=0.56)



    # test_sine()
    # test_polynom()
    # test_liouville()
    # test_ldf()
    # test_bsf()
    # test_bowshock()



