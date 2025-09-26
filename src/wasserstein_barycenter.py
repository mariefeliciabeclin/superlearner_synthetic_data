
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

def common_grid(df_list, n_grid =1000):
    """Compute a common grid for a set of dataframes

    Args:
        df_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    #d = len(pd.DataFrame(df_list[0]).columns)
    d = np.shape(df_list[0])[1]
    print(d)
    grid_hist = []
    for i in range(d):
        min = np.array([np.array(df)[:,i].min() for df in df_list]).min()
        max = np.array([np.array(df)[:,i].max() for df in df_list]).max()
        grid_hist.append(np.linspace(min, max,  n_grid))
    return grid_hist



def dataframes_in_hists(df_list):
    """ Compute histogramme of a data frame

    Args:
        df_list (_type_): list or array of dataframes

    Returns:
        _type_: _description_
    """
    grid_hist = common_grid(df_list)
    return np.array([np.histogramdd(df, bins= grid_hist)[0] for df in df_list])
            
            

def wasserstein_barycenter(df_list, weights,  cost_fns: Sequence[costs.CostFn]):
    """_summary_

    Args:
        df_list (_list_): List of dataframe, each dataframe shall have the same variables
    """
    hists =  dataframes_in_hists(df_list)
    grid_size = hists.shape[1:3]

# Create the grid, with the cost
    g_grid = grid.Grid(
        x=[jnp.arange(0, n) / 100 for n in grid_size],
        cost_fns=cost_fns,
        epsilon=epsilon_scheduler.Epsilon(target=1e-4, init=1e-1, decay=0.95),
    )
    hists = hists.reshape((3, -1)) + 1e-2

    hists = hists / np.sum(hists, axis=1)[:, np.newaxis]
## Wasserstein barycenter with weights [0.1, 0.6, 0.3]
    solver = jax.jit(db.FixedBarycenter())
    problem = bp.FixedBarycenterProblem(g_grid, a, weights= weights)
    barycenter = solver(problem)

    return(barycenter)

    
def sample_in_barycenter(barycenter, n_sample):
    bary = barycenter.histogram
    probabilities = bary /bary.sum()
    probabilities = np.array(bary /bary.sum())
    probabilities[-1] += 1 - probabilities.sum()
    selected_bin = np.random.choice(bin_indices,size=n_sample, replace=True, p=probabilities)
    return selected_bin

        
def aggregation_by_wasserstein_barycenter(df_list,  weights, n_sample):
    barycenter = wasserstein_barycenter(df_list, weights)
    selected_bin = sample_in_barycenter(barycenter=barycenter, n_sample=n_sample)
    return 


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

print('ok')

X = random.multivariate_normal(mean=mean_X, cov=cov_X, size=n_X)
Y = random.multivariate_normal(mean=mean_Y, cov=cov_Y, size=n_Y)
Z = random.multivariate_normal(mean=mean_Z, cov=cov_Z, size=n_Z)

distribs = [X, Y, Z]

print('ok')
barycenter = wasserstein_barycenter(distribs,[0.1, 0.6, 0.3],[L1Distance] )
print('ok')
sample_in_barycenter(barycenter=barycenter, n_sample =1000)

print(barycenter)
print(sample_in_barycenter)
    