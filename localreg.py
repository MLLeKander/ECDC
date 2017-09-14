from kernel import kernels
from args import *
import numpy as np
from utils import PrettyDeque

arg_parser.add_argument('--tree_consistency_iters', type=int, default=10)
arg_parser.add_argument('--kernel', choices=kernels.keys(), default='constant')

class LocalReg(object):
    def __init__(self, k, nn_forest, kernel=None, tree_consistency_iters=None, drift_hist_len=6, drift_thresh=20):
        if kernel is None:
            kernel = args.kernel
        if tree_consistency_iters is None:
            tree_consistency_iters = args.tree_consistency_iters

        self.k = k
        self.forest = nn_forest
        self.tree_consistency_iters = tree_consistency_iters

        self.kernel = kernels[kernel] if isinstance(kernel, str) else kernel

    def size(self):
        return self.forest.size()

    def add(self, data, label):
        self.forest.add(data, label)
        for i in range(self.tree_consistency_iters):
            self.forest.enforce_tree_consistency_random()

    def enforce_drift(self, data, target, forest_results):
        pass

    def query(self, X):
        dists, labels, ndxes, data = self.forest.neighbors(X, self.k)
        if len(dists) == 0:
            return (np.nan, None)

        weights = self.kernel(dists)
        forest_results = (weights, dists, labels, ndxes, data)
        return self._predict(X, *forest_results), forest_results

    def _predict(self, X, weights, dists, labels, ndxes, data):
        raise NotImplementedError

class LocalConstantReg(LocalReg):
    def _predict(self, X, weights, dists, labels, ndxes, data):
        return np.sum(weights*labels)/np.sum(weights)

class LocalLinearReg(LocalReg):
    def _predict(self, X, weights, dists, labels, ndxes, data):
        w_sqrt = np.sqrt(weights)
        fit = np.linalg.lstsq(X * w_sqrt[:,None], labels * w_sqrt)[0]
        return fit.dot(X)
