import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'


# matplotlib.rcParams['lines.linewidth'] = 0.5


def plot_mesh(pos, edge_index, show_idx=False, title=None, saving=None, **kwargs):
    # pos: array, shape=[n, 2]
    # edge_index, array, shape=[2, E]
    figsize = kwargs.get('figsize', (4, 4))
    pos_src = pos[edge_index[0]]
    pos_tag = pos[edge_index[1]]
    lines = np.stack((pos_src, pos_tag), axis=1)
    fig, ax = plt.subplots(figsize=figsize)
    lc = LineCollection(lines, linewidths=0.5, colors='blue', alpha=0.7)
    ax.add_collection(lc)
    if show_idx:
        for idx, (x, y) in enumerate(pos):
            ax.text(x, y, str(idx), color='red', fontsize=10, ha='left', va='bottom')
    x_min, y_min = pos.min(axis=0)
    x_max, y_max = pos.max(axis=0)
    padding = 0.05
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal')
    # ax.grid(True, linestyle='--', alpha=0.3)
    if title is not None:
        ax.set_title(title)
    if saving is not None:
        plt.savefig(saving, dpi=300, bbox_inches='tight')
    plt.show()


def plot_regular_contour2D(pos, u, cmap='viridis', w=None, h=None, title=None, saving=None, **kwargs):
    # pos: array (shape=[n, 2]) or tuple/list of arrays ((x_ij, y_ij), shape=[w+1, h+1]), where n=(w+1)*(h+1)
    # u: array (shape=[n, ] or [w+1, h+1]
    # w, h: int, number of grid cells in axis of x, y
    if isinstance(pos, np.ndarray) and (w is None or h is None):
        raise ValueError('Please provide w and h for reshape of 1D flat array!')
    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    shrink = kwargs.get('shrink', 1.)
    figsize = kwargs.get('figsize', (4.75, 4))
    plt.figure(figsize=figsize)
    ax = plt.gca()
    if isinstance(pos, (tuple, list)):
        x_ij, y_ij = pos
    elif isinstance(pos, np.ndarray):
        x_ij = pos[:, 0].reshape(h + 1, w + 1)
        y_ij = pos[:, 1].reshape(h + 1, w + 1)
        u = u.reshape(h + 1, w + 1)
    else:
        raise TypeError(f'Input pos must be array, tuple or list, got {type(pos).__name__}!')
    u_min, u_max = np.min(u), np.max(u)
    ax.set_xlim(x_ij.min(), x_ij.max())
    ax.set_ylim(y_ij.min(), y_ij.max())
    levels = np.linspace(u_min, u_max, 20)
    contourf = plt.contourf(x_ij, y_ij, u, levels=levels, cmap=cmap, vmin=u_min, vmax=u_max)
    cbar = plt.colorbar(contourf, shrink=shrink, aspect=20, pad=0.02)
    # cbar.locator = MaxNLocator(nbins=5)
    cbar.locator = plt.FixedLocator(np.linspace(u_min, u_max, 5))
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e' if np.max(np.abs(u)) < 1e-2 else '%.2f'))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')
    if x_label is not None:
        plt.xlabel(x_label, fontsize=10)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=10)
    if saving is not None:
        plt.savefig(saving, dpi=300, bbox_inches='tight')
    plt.show()


def plot_irregular_contour2D(pos, u, triangles=None, title=None, cmap='viridis', saving=None, **kwargs):
    # pos: array, shape=[n, 2]
    # u: array, shape=[n, ]
    # triangles: triangle ids, array, shape=[n, 3], optional, if not given, use automatic Delaunay Triangulation
    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    shrink = kwargs.get('shrink', 0.95)
    padding = kwargs.get('padding', 0.05)
    figsize = kwargs.get('figsize', (4.75, 4))
    x, y = pos[:, 0], pos[:, 1]
    tris = tri.Triangulation(x, y, triangles=triangles if triangles is not None else None)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_xlim(x.min() - padding, x.max() + padding)
    ax.set_ylim(y.min() - padding, y.max() + padding)
    u_min, u_max = np.min(u), np.max(u)
    # levels = np.linspace(u_min, u_max, 30)
    # contourf = ax.tricontourf(tris, u, levels=levels, cmap=cmap, vmin=u_min, vmax=u_max)
    if u_max - u_min < 1e-8:
        contourf = ax.tricontourf(tris, u, cmap=cmap)
    else:
        levels = np.linspace(u_min, u_max, 30)
        contourf = ax.tricontourf(tris, u, levels=levels, cmap=cmap, vmin=u_min, vmax=u_max)
    cbar = plt.colorbar(contourf, shrink=shrink, aspect=30, pad=0.02)
    cbar.locator = plt.FixedLocator(np.linspace(u_min, u_max, 5))
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e' if np.max(np.abs(u)) < 1e-2 else '%.2f'))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')
    if x_label is not None:
        plt.xlabel(x_label, fontsize=10)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=10)
    if saving is not None:
        plt.savefig(saving, dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close()


def plot_irregular_scatter2D(pos, u, title=None, cmap='jet', saving=None, **kwargs):
    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    shrink = kwargs.get('shrink', 0.95)
    padding = kwargs.get('padding', 0.05)
    figsize = kwargs.get('figsize', (5, 4))
    marker_size = kwargs.get('marker_size', 5)
    x, y = pos[:, 0], pos[:, 1]
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_xlim(x.min() - padding, x.max() + padding)
    ax.set_ylim(y.min() - padding, y.max() + padding)
    u_min, u_max = np.min(u), np.max(u)
    scatter = ax.scatter(x, y, c=u, cmap=cmap, s=marker_size,
                         vmin=u_min, vmax=u_max, edgecolors='none')
    cbar = plt.colorbar(scatter, shrink=shrink, aspect=30, pad=0.02)
    cbar.locator = plt.FixedLocator(np.linspace(u_min, u_max, 5))
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.2e' if np.max(np.abs(u)) < 1e-2 else '%.2f')
    )
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')
    if x_label is not None:
        plt.xlabel(x_label, fontsize=10)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=10)
    if saving is not None:
        plt.savefig(saving, dpi=300, bbox_inches='tight')
    plt.show()


def plot_line_chart(vars, x=None, x_label=None, y_label=None, title=None, saving=None, **kwargs):
    # vars: array or tuple/list of arrays, shape=[n, ]
    plt.figure(figsize=(6, 4))
    colors = kwargs.get('colors', None)
    labels = kwargs.get('labels', None)
    if isinstance(vars, np.ndarray):
        x_data = np.arange(vars.shape[0]) if x is None else x
        if x is None: plt.xticks(x_data)
        if colors is not None:
            plt.plot(x_data, vars, colors=colors)
        else:
            plt.plot(x_data, vars)
    elif isinstance(vars, (list, tuple)):
        x_data = np.arange(vars[0].shape[0]) if x is None else x
        if x is None: plt.xticks(x_data)
        for i, var in enumerate(vars):
            if colors is not None:
                if labels is not None:
                    plt.plot(x_data, var, color=colors[i], label=labels[i])
                else:
                    plt.plot(x_data, var, color=colors[i])
            else:
                plt.plot(x_data, var)
    else:
        raise TypeError(f'Input vars must be array, tuple or list, got {type(vars).__name__}!')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')
    if x_label is not None:
        plt.xlabel(x_label, fontsize=10)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=10)
    if labels is not None:
        plt.legend(fontsize=9)
    if saving is not None:
        plt.savefig(saving, dpi=300, bbox_inches='tight')
    plt.show()

