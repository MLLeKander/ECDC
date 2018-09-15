from driftstrategy import default_drift_strategy
from kernel import kernels
from args import *
import numpy as np
from utils import PrettyDeque
import scipy.stats

arg_group = arg_parser.add_argument_group('regressor arguments')
arg_group.add_argument('--kernel', choices=kernels.keys(), default='constant')
arg_group.add_argument('--tree_consistency_iters', type=int, default=10)
arg_group.add_argument('--match_exact', type=str2bool, default=True)

class LocalReg(object):
    @clidefault
    def __init__(self, k, nn_forest, kernel=CLIArg, tree_consistency_iters=CLIArg, match_exact=CLIArg):
        self.k = k
        self.forest = nn_forest

        self.kernel = kernels[kernel] if isinstance(kernel, str) else kernel
        self.tree_consistency_iters = tree_consistency_iters
        self.match_exact = match_exact

        self.evict_count = 0

        self.drift_strategy = default_drift_strategy(self)

    def size(self):
        return self.forest.size()

    def add(self, X, label):
        old_tail_ndx = self.forest.get_tail_ndx()

        del_ndx = self.forest.add(X, label)
        head_ndx = self.forest.get_head_ndx()

        self.drift_strategy.add(X, label, old_tail_ndx, del_ndx, head_ndx)

        for i in range(self.tree_consistency_iters):
            self.forest.enforce_tree_consistency_random()
        return head_ndx

    def clear(self, ndx):
        old_ndx = self.forest.get_tail_ndx()
        old_ndx2 = self.forest.clear(ndx)
        assert(old_ndx == old_ndx2)
        self.evict_count += 1
        return old_ndx

    def query(self, X):
        if self.match_exact:
            exact_ndx = self.forest.lookup_exact(X)
            if self.forest.is_valid_ndx(exact_ndx):
                return (self.forest.get_label(exact_ndx), exact_ndx)
        dists, labels, ndxes, data = self.forest.neighbors(X, self.k)
        if len(dists) == 0:
            return (np.nan, None)

        weights = self.kernel(dists)
        prediction = self._predict(X, weights, dists, labels, ndxes, data)
        forest_results = (weights, dists, labels, ndxes, data, prediction)
        return prediction, forest_results

    def query_exact(self, X):
        exact_ndx = self.forest.lookup_exact(X)
        if self.forest.is_valid_ndx(exact_ndx):
            return self.forest.get_label(exact_ndx)
        else:
            return None

    def update_drift(self, X, target, forest_results):
        self.drift_strategy.update_drift(X, target, forest_results)

    def enforce_drift(self):
        self.drift_strategy.enforce_drift()

    def _leave_one_out_predictions(self, X, weights, dists, labels, ndxes, data):
        hats = []
        for sub_ndx in range(len(dists)):
            sub_weights = np.delete(weights, sub_ndx, axis=0)
            sub_dists = np.delete(dists, sub_ndx, axis=0)
            sub_labels = np.delete(labels, sub_ndx, axis=0)
            sub_ndxes = np.delete(ndxes, sub_ndx, axis=0)
            sub_data = np.delete(data, sub_ndx, axis=0)
            hats.append(self._predict(X, sub_weights, sub_dists, sub_labels, sub_ndxes, sub_data))
        return hats

    def _predict(self, X, weights, dists, labels, ndxes, data):
        raise NotImplementedError

class LocalConstantReg(LocalReg):
    def _predict(self, X, weights, dists, labels, ndxes, data):
        return np.sum(weights*labels)/np.sum(weights)

    def _leave_one_out_predictions(self, X, weights, dists, labels, ndxes, data):
        if len(weights) == 0:
            return np.nan

        numerator = np.sum(weights*labels)
        denominator = np.sum(weights)

        hats = []
        for weight, label in zip(weights, labels):
            hat = (numerator-weight*label)/(denominator-weight)
            hats.append(hat)
        return hats

class LocalLinearReg(LocalReg):
    def _predict(self, X, weights, dists, labels, ndxes, data):
        w_sqrt = np.sqrt(weights)
        data_with_bias = np.insert(data, 0, 1, axis=1)
        fit = np.linalg.lstsq(data_with_bias * w_sqrt[:,None], labels * w_sqrt)[0]
        return fit.dot(np.insert(X, 0, 1))
