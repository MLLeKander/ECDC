from kernel import kernels
from args import *
import numpy as np
from utils import PrettyDeque
import scipy.stats

arg_parser.add_argument('--kernel', choices=kernels.keys(), default='constant')
arg_parser.add_argument('--tree_consistency_iters', type=int, default=10)
arg_parser.add_argument('--match_exact', type=str2bool, default=True)
arg_parser.add_argument('--drift_hist_len', type=int, default=-1)
arg_parser.add_argument('--drift_thresh', type=float, default=-1)

class LocalReg(object):
    def __init__(self, k, nn_forest, kernel=None, tree_consistency_iters=None, match_exact=None, drift_hist_len=None, drift_thresh=None):
        if kernel is None:
            kernel = args.kernel
        if tree_consistency_iters is None:
            tree_consistency_iters = args.tree_consistency_iters
        if match_exact is None:
            match_exact = args.match_exact
        if drift_hist_len is None:
            drift_hist_len = args.drift_hist_len
        if drift_thresh is None:
            drift_thresh = args.drift_thresh

        self.k = k
        self.forest = nn_forest

        self.kernel = kernels[kernel] if isinstance(kernel, str) else kernel
        self.tree_consistency_iters = tree_consistency_iters
        self.match_exact = match_exact
        self.drift_hist_len = drift_hist_len
        self.drift_thresh = drift_thresh

        #TODO: Type will need to be larger for higher k > 128
        if (self.drift_hist_len > 0) != (self.drift_thresh > 0):
            raise ValueError('Either both or neither of drift_hist_len and drift_thresh must be specified')
        elif self.drift_hist_len > 0:
            self.drift_hist = np.full((nn_forest.get_memory_size(), drift_hist_len), -1, dtype=np.int8)
            self.active_ndxes = set()
        else:
            self.drift_hist = None
            self.active_ndxes = None
        self.evict_count = 0

    def size(self):
        return self.forest.size()

    def add(self, X, label):
        old_tail_ndx = self.forest.get_tail_ndx()

        del_ndx = self.forest.add(X, label)
        head_ndx = self.forest.get_head_ndx()

        self._drift_hist_new(old_tail_ndx, del_ndx, head_ndx)

        for i in range(self.tree_consistency_iters):
            self.forest.enforce_tree_consistency_random()
        return head_ndx

    def clear(self, ndx):
        old_ndx = self.forest.get_tail_ndx()
        old_ndx2 = self.forest.clear(ndx)
        assert(old_ndx == old_ndx2)
        if self.drift_hist is not None:
            if old_ndx != ndx:
                self.drift_hist[ndx, :] = self.drift_hist[old_ndx, :]
            self.drift_hist[old_ndx, :] = np.full((self.drift_hist_len,), -1, dtype=np.int8)

    def _drift_hist_new(self, old_tail_ndx, del_ndx, head_ndx):
        if self.drift_hist is None:
            return
        #TODO: This may not work for (near-)empty cases
        new_drift_hist = np.full((self.drift_hist_len,), 0, dtype=np.int8)
        if self.forest.get_memory_size() != del_ndx:
            new_drift_hist = self.drift_hist[del_ndx, :].copy()
            if del_ndx != old_tail_ndx:
                self.drift_hist[del_ndx, :] = self.drift_hist[old_tail_ndx, :]
            self.drift_hist[old_tail_ndx, :] = -1
        self.drift_hist[head_ndx] = new_drift_hist

    def _drift_hist_add(self, ndx, rank):
        if self.drift_hist is None:
            pass
        if self.drift_hist[ndx,0] == -1:
            raise ValueError('Attempt to add to invalid drift_hist ndx')
        tmp = np.roll(self.drift_hist[ndx,:],1)
        tmp[0] = rank
        self.drift_hist[ndx,:] = tmp
        if rank < self.drift_thresh:
            self.active_ndxes.add(ndx)

    def query(self, X):
        if self.match_exact:
            exact_ndx = self.forest.lookup_exact(X)
            if self.forest.is_valid_ndx(exact_ndx):
                return (self.forest.get_label(exact_ndx), None)
        dists, labels, ndxes, data = self.forest.neighbors(X, self.k)
        if len(dists) == 0:
            return (np.nan, None)

        weights = self.kernel(dists)
        prediction = self._predict(X, weights, dists, labels, ndxes, data)
        forest_results = (weights, dists, labels, ndxes, data, prediction)
        return prediction, forest_results

    def update_drift(self, X, target, forest_results):
        if self.drift_hist is None:
            return

        if forest_results is None:
            return
        elif isinstance(forest_results, int):
            dists, labels, ndxes, data = self.forest.neighbors(X, self.k)
            if len(dists) == 0:
                return
            weights = self.kernel(dists)
            prediction = self._predict(X, weights, dists, labels, ndxes, data)
            exact_ndx = forest_results
        else:
            weights, dists, labels, ndxes, data, prediction = forest_results
            exact_ndx = None

        if len(dists) <= 1:
            return

        hats = np.array(self._leave_one_out_predictions(X, weights, dists, labels, ndxes, data))
        hat_errs = np.abs(hats - target)

        ranks = scipy.stats.rankdata(hat_errs, method='min').astype(np.int8)
        if exact_ndx is None:
            for ndx, rank in zip(ndxes, ranks):
                self._drift_hist_add(ndx, rank)
        elif exact_ndx in ndxes:
            exact_sub_ndx = nonzero(ndxes==exact_ndx)[0][0]
            self._drift_hist_add(exact_ndx, ranks[exact_sub_ndx])
        else:
            print 'exact_ndx not found in ndxes'

    def enforce_drift(self):
        if self.drift_hist is None:
            return
        for i in self.active_ndxes:
            while self.drift_hist[i, :].mean() > self.drift_thresh:
                self.evict_count += 1
                self.clear(i)
        self.active_ndxes.clear()

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
        numerator = np.sum(weights*labels)
        denominator = np.sum(weights)

        hats = []
        for weight, label in zip(weights, labels):
            hat = (numerator-weight*label)/(denominator-weight*label)
            hats.append(hat)
        return hats

class LocalLinearReg(LocalReg):
    def _predict(self, X, weights, dists, labels, ndxes, data):
        w_sqrt = np.sqrt(weights)
        data_with_bias = np.insert(data, 0, 1, axis=1)
        fit = np.linalg.lstsq(data_with_bias * w_sqrt[:,None], labels * w_sqrt)[0]
        return fit.dot(np.insert(X, 0, 1))
