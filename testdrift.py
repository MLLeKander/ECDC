from localreg import *
from args import *
from vqtree import KForest
dims = 1
size = 5

np.random.seed(2)

parse_args()

forest = KForest(dims, size*2, remove_dups=True)

reg = LocalReg(5, forest)

def print_meta():
    global reg
    for ndx, row in enumerate(reg.drift_hist):
        print ndx, row

data = np.random.randn(size*2, dims)
for i in range(len(data)):
    data[i,0] = i

for ndx, d in enumerate(data):
    new_loc = reg.add(d, ndx)
    #reg._add_to_drift_hist(new_loc, ndx)
#print_meta()


count = len(data)

for _ in range(size*4):
    ndx = np.random.randint(size)
    new_loc = reg.add(data[ndx], count)
    #reg._add_to_drift_hist(new_loc, count)
    count += 1
    #print_meta()
    forest.print_tree()
