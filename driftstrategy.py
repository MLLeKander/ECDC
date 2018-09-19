from args import *
import numpy as np
import scipy.stats

def default_drift_strategy(local_reg):
    return strategies[args.drift_strategy](local_reg)

class DriftStrategy(object):
    def __init__(self, local_reg):
        self.local_reg = local_reg
        self.drift_hist = None
    def add(self, X, label, old_tail_ndx, del_ndx, head_ndx):
        pass
    def update_drift(self, X, target, forest_results):
        raise NotImplementedError()
    def enforce_drift(self):
        raise NotImplementedError()

class NoDrift(DriftStrategy):
    def update_drift(self, X, target, forest_results):
        pass
    def enforce_drift(self):
        pass

class NeighborDriftStrategy(DriftStrategy):
    @clidefault
    def __init__(self, local_reg, drift_exact=CLIArg, drift_random=CLIArg):
        super(NeighborDriftStrategy, self).__init__(local_reg)
        self.drift_exact = drift_exact
        self.drift_random = drift_random
        self.active_ndxes = set()

    def update_drift(self, X, target, forest_results):
        if forest_results is None:
            return
        elif isinstance(forest_results, int):
            if not self.drift_exact:
                return
            else:
                dists, labels, ndxes, data = self.local_reg.forest.neighbors(X, self.local_reg.k)
                if len(dists) == 0:
                    return
                weights = self.local_reg.kernel(dists)
                prediction = self.local_reg._predict(X, weights, dists, labels, ndxes, data)
                exact_ndx = forest_results
        else:
            weights, dists, labels, ndxes, data, prediction = forest_results
            exact_ndx = None

        if len(dists) <= 1:
            return

        self._update_drift(X, target, weights, dists, labels, ndxes, data, prediction, exact_ndx)

    def clear_ndx(self, ndx):
        return self.local_reg.clear(ndx)

    def clear_ndxes(self, ndxes):
        ndxes_sorted = sorted(ndxes, key=lambda x: (x < self.local_reg.forest.get_tail_ndx(), x))
        for ndx in ndxes_sorted:
            if np.random.rand() <= self.drift_random:
                self.clear_ndx(ndx)

    def _update_drift(self, X, target, weights, dists, labels, ndxes, data, prediction, exact_ndx):
        raise NotImplementedError()

class RandomEvict(NeighborDriftStrategy):
    def _update_drift(self, X, target, weights, dists, labels, ndxes, data, prediction, exact_ndx):
        self.active_ndxes.update(ndxes)

    def enforce_drift(self):
        self.clear_ndxes(self.active_ndxes)
        #to_clear = np.random.rand(len(self.active_ndxes)) < self.drift_random
        #evict_ndxes = np.array(list(self.active_ndxes))[to_clear]
        #self.clear_ndxes(evict_ndxes)

        self.active_ndxes.clear()

class LeaveOneOut(NeighborDriftStrategy):
    @clidefault
    def __init__(self, local_reg, drift_hist_len=CLIArg, drift_thresh=CLIArg, **kwargs):
        super(LeaveOneOut, self).__init__(local_reg, **kwargs)
        self.drift_hist_len = drift_hist_len
        self.drift_thresh = drift_thresh
        self.drift_hist = self._drift_hist_init(self.local_reg.forest.get_memory_size())

    def _drift_hist_new(self, old_tail_ndx, del_ndx, head_ndx):
        #TODO: This may not work for (near-)empty cases
        if self.local_reg.forest.get_memory_size() != del_ndx:
            new_drift_hist = self.drift_hist[del_ndx, :].copy()
            if del_ndx != old_tail_ndx:
                if not self.local_reg.forest.is_valid_ndx(del_ndx):
                    raise ValueError('_drift_hist_new called with invalid del_ndx')
                self.drift_hist[del_ndx, :] = self.drift_hist[old_tail_ndx, :]
        else:
            new_drift_hist = self._drift_hist_init()

        if not self.local_reg.forest.is_valid_ndx(head_ndx):
            raise ValueError('_drift_hist_new called with invalid head_ndx')
        self.drift_hist[head_ndx, :] = new_drift_hist

    def _drift_hist_add(self, ndx, stat):
        if not self.local_reg.forest.is_valid_ndx(ndx):
            raise ValueError('Attempt to add to invalid drift_hist ndx')
        tmp = np.roll(self.drift_hist[ndx,:],1)
        tmp[0] = stat
        self.drift_hist[ndx,:] = tmp

    def add(self, X, label, old_tail_ndx, del_ndx, head_ndx):
        self._drift_hist_new(old_tail_ndx, del_ndx, head_ndx)

    def clear_ndx(self, ndx):
        old_ndx = super(LeaveOneOut, self).clear_ndx(ndx)
        if old_ndx != ndx:
            self.drift_hist[ndx, :] = self.drift_hist[old_ndx, :]
        #TODO: check shape is okay
        self.drift_hist[old_ndx, :] = self._drift_hist_init()
        #self.drift_hist[old_ndx, :] = self._drift_hist_init((self.drift_hist_len,))
        return old_ndx

    def _update_drift(self, X, target, weights, dists, labels, ndxes, data, prediction, exact_ndx):
        yhat_err = np.abs(target-prediction)
        ytildes = np.array(self.local_reg._leave_one_out_predictions(X, weights, dists, labels, ndxes, data))
        ytilde_errs = np.abs(target-ytildes)
        loo_stats = self._loo_stats(yhat_err, ytilde_errs)

        if exact_ndx is not None and exact_ndx not in ndxes:
            print 'exact_ndx not found in ndxes'
        else:
            for ndx, stat in zip(ndxes, loo_stats):
                if exact_ndx is None or ndx == exact_ndx:
                    self._drift_hist_add(ndx, stat)
                    if self.drift_hist[ndx,:].mean() >= self.drift_thresh:
                        self.active_ndxes.add(ndx)

    def enforce_drift(self):
        del_ndxes = [i for i in self.active_ndxes if self.drift_hist[i,:].mean() >= self.drift_thresh]
        self.clear_ndxes(del_ndxes)
        self.active_ndxes.clear()

    def _drift_hist_init(self, length=1):
        #return np.full((length, self.drift_hist_len), 0, dtype=np.int8)
        raise NotImplementedError()

    # high return = lower ytilde err
    def _loo_stats(self, yhat_err, ytilde_errs):
        raise NotImplementedError()

class LOO_Binary(LeaveOneOut):
    def _drift_hist_init(self, length=1):
        return np.full((length, self.drift_hist_len), False, dtype=np.bool)

    def _loo_stats(self, yhat_err, ytilde_errs):
        return ytilde_errs + 1e-6 < yhat_err

class LOO_Rank(LeaveOneOut):
    def _drift_hist_init(self, length=1):
        return np.full((length, self.drift_hist_len), -1, dtype=np.int8)

    def _loo_stats(self, yhat_err, ytilde_errs):
        return scipy.stats.rankdata(-ytilde_errs, method='min').astype(np.int8)

class LOO_Rank_Bad(LeaveOneOut):
    @clidefault
    def __init__(self, local_reg, drift_bad_errrank=CLIArg, drift_bad_activesign=CLIArg, **kwargs):
        super(LOO_Rank_Bad, self).__init__(local_reg, **kwargs)
        self.drift_bad_errrank = drift_bad_errrank
        self.drift_bad_activesign = drift_bad_activesign

    def _drift_hist_init(self, length=1):
        return np.full((length, self.drift_hist_len), 0, dtype=np.int8)

    def _loo_stats(self, yhat_err, ytilde_errs):
        if self.drift_bad_errrank:
            return scipy.stats.rankdata(ytilde_errs, method='min').astype(np.int8)
        else:
            return scipy.stats.rankdata(-ytilde_errs, method='min').astype(np.int8)

    def _update_drift(self, X, target, weights, dists, labels, ndxes, data, prediction, exact_ndx):
        yhat_err = np.abs(target-prediction)
        ytildes = np.array(self.local_reg._leave_one_out_predictions(X, weights, dists, labels, ndxes, data))
        ytilde_errs = np.abs(target-ytildes)
        loo_stats = self._loo_stats(yhat_err, ytilde_errs)

        if exact_ndx is not None and exact_ndx not in ndxes:
            print 'exact_ndx not found in ndxes'
        else:
            for ndx, stat in zip(ndxes, loo_stats):
                if exact_ndx is None or ndx == exact_ndx:
                    self._drift_hist_add(ndx, stat)
                    if self.drift_bad_activesign and stat < self.drift_thresh:
                        self.active_ndxes.add(ndx)
                    elif not self.drift_bad_activesign and stat >= self.drift_thresh:
                        self.active_ndxes.add(ndx)

strategies = {
    'none': NoDrift,
    'random': RandomEvict,
    'loobinary': LOO_Binary,
    'loorank': LOO_Rank,
    'loorankbad': LOO_Rank_Bad,
}

arg_group = arg_parser.add_argument_group('drift arguments')
arg_group.add_argument('--drift_strategy', choices=strategies.keys(), default='none', help='')
arg_group.add_argument('--drift_exact', type=str2bool, default=False)
arg_group.add_argument('--drift_random', type=float, default=1)
arg_group.add_argument('--drift_hist_len', type=int, default=10)
arg_group.add_argument('--drift_thresh', type=float, default=0)
arg_group.add_argument('--drift_bad_errrank', type=str2bool, default=True)
arg_group.add_argument('--drift_bad_activesign', type=str2bool, default=True)
