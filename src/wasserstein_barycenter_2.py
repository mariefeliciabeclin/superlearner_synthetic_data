from typing import Sequence
from jax import jit
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from ott.geometry import costs, epsilon_scheduler, grid
from ott.problems.linear import barycenter_problem as bp
from ott.solvers.linear import discrete_barycenter as db
from ott.geometry import costs
import numpy.random as random
import numpy as np


@register_pytree_node_class
class L1Distance(costs.CostFn):
    """Computes the L1 distance between x and y."""

    def __call__(self, x: np.ndarray, y: np.ndarray):
        return jnp.sum(jnp.abs(x - y) ** 2)


def prepare_bins(distribs: list[np.ndarray]) -> list[np.ndarray]:
    bins = []
    for i in range(distribs[0].shape[-1]):
        distrib_i_concat = np.concatenate([distrib[:, i] for distrib in distribs])
        i_min = distrib_i_concat.min()
        i_max = distrib_i_concat.max()

        i_bins = np.linspace(i_min, i_max, 100)
        bins.append(i_bins)

    return bins


def prepare_hists(distribs: list[np.ndarray], bins: list[np.ndarray]):
    hists = []
    for distrib in distribs:
        hist, _ = np.histogramdd(
            np.column_stack([distrib[:, i] for i in range(distrib.shape[-1])]),
            bins=bins,
        )

        hists.append(hist)
    return hists


## Custom cost for Wasserstein distance ( euclidean cost here)
def calculate_barycenter(
    distribs: list[np.ndarray], weights: np.ndarray, cost_fns: Sequence[costs.CostFn]
):

    bins = prepare_bins(distribs)

    hists = prepare_hists(distribs, bins)

    #### I don't know how to name a array
    a = np.array(hists)

    grid_size = a.shape[1:]

    # Create the grid, with the cost
    g_grid = grid.Grid(
        x=[jnp.arange(0, n) / 100 for n in grid_size],
        cost_fns=cost_fns,
        epsilon=epsilon_scheduler.Epsilon(target=1e-4, init=1e-1, decay=0.95),
    )

    ## Reshape the list of distributions
    a = a.reshape((a.shape[0], -1)) + 1e-2

    a = a / np.sum(a, axis=1)[:, np.newaxis]

    ## Wasserstein barycenter with weights [0.1, 0.6, 0.3]
    solver = jit(db.FixedBarycenter())
    problem = bp.FixedBarycenterProblem(g_grid, a, weights=weights)
    barycenter = solver(problem)

    return barycenter


# RUN

print("ok")

n_X = 500
mean_X = [1, 2, 3]
cov_X = [[0.5, 0, 0], [0, 5.0, 0], [0, 0, 0.5]]

n_Y = 200
mean_Y = [1, 3, 3]
cov_Y = [[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5]]

n_Z = 300
mean_Z = [0, 0, 0]
cov_Z = cov_Z = [[0.2, 0.05, 0.03], [0.05, 0.3, 0.07], [0.03, 0.07, 0.4]]

print("ok")

X = random.multivariate_normal(mean=mean_X, cov=cov_X, size=n_X)
Y = random.multivariate_normal(mean=mean_Y, cov=cov_Y, size=n_Y)
Z = random.multivariate_normal(mean=mean_Z, cov=cov_Z, size=n_Z)

print("ok")

distribs = [X, Y, Z]
weights = np.array([0.1, 0.6, 0.3])

print("ok")

barycenter = calculate_barycenter(distribs, weights, [L1Distance()])
print("ok")

print(barycenter)