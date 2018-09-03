from localreg import *
from args import *
from vqtree import KForest
k = 10
epoch = 400
np.random.seed(2)
parse_args()

forest = KForest(dim=1, memory_size=epoch*10, remove_dups=True)

#reg = LocalConstantReg(k, forest, match_exact=False, drift_hist_len=5, drift_thresh=k*1.5/2)
#reg = LocalConstantReg(k, forest, match_exact=False, drift_hist_len=10, drift_thresh=6)
reg = LocalConstantReg(k, forest, match_exact=False, drift_hist_len=5, drift_thresh=9)

def print_meta(forest_results, X, target):
    global reg, forest
    acnt, bcnt = 0, 0
    for i in range(forest.get_memory_size()):
        if forest.is_valid_ndx(i):
            if forest.get_data(i)[0] == forest.get_label(i):
                acnt += 1
            else:
                bcnt += 1
    print 'A:%-4d B:%-4d'%(acnt,bcnt)
    if forest_results is not None:
        weights, dists, labels, ndxes, data, prediction = forest_results
        preds = reg._leave_one_out_predictions(X, weights, dists, labels, ndxes, data)
        errs = np.abs(target-np.array(preds))
        for dat, label, ndx, err in zip(data, labels, ndxes, errs):
            #print '%4d(%d): %5.3f %s'%(ndx, (label < 0) ==  (dat[0] < 0), pred, reg.drift_hist[ndx])
            print '%4d(%5.0f -> %5.0f): %5.0f %4.1f %s'%(ndx, label, dat[0], err, reg.drift_hist[ndx].mean(), reg.drift_hist[ndx])
        if len(reg.active_ndxes) > 0:
            print 'Deleting', reg.active_ndxes
    #for ndx, row in enumerate(reg.drift_hist):
    #    print ndx, row

def run_epoch(fun):
    data = (np.random.random((epoch,1))*20-10)*1000
    for x in data:
        target = fun(x)
        q, forest_results = reg.query(x)
        print 'X:%-5.0f Query:%-5.0f'%(x[0], q)
        reg.update_drift(x, target, forest_results)
        print_meta(forest_results, x, target)
        reg.enforce_drift()
        #if reg.evict_count > 0:
        #    print 'Something wrong:', i, x
        #    reg.evict_count = 0
        reg.add(x, target)


run_epoch(lambda x : -x[0])
run_epoch(lambda x : x[0])
