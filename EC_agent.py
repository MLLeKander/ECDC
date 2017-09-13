import gym
import random
import numpy as np
from args import *
from localreg import LocalConstantReg, LocalLinearReg
from vqtree import KForest
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

arg_parser.add_argument('--num_trees', type=int, default=2)
arg_parser.add_argument('--memory_size', type=int, default=500000)
arg_parser.add_argument('--spill', type=float, default=0.1)
arg_parser.add_argument('--eps', type=float, default=0.005)
arg_parser.add_argument('--max_dims', type=int, default=64)
arg_parser.add_argument('--k', type=int, default=11)
arg_parser.add_argument('--regressor_type', choices=['kernel','linear'], default='kernel')
arg_parser.add_argument('--dry_run', type=str2bool, default=False)

class FlattenProjection(object):
    def __init__(self, in_shape):
        self.in_shape = in_shape

    def out_dims(self):
        return np.prod(self.in_shape)

    def __call__(self,in_vec):
        return in_vec.ravel()

class RandomProjection(object):
    def __init__(self, in_shape, out_dims):
        self.projection = SparseRandomProjection(n_components=out_dims)
        num_dims = np.prod(in_shape)
        self.projection.fit([[1]*num_dims])
        self._out_dims = out_dims

    def out_dims(self):
        return self._out_dims

    def __call__(self, in_vec):
        # Ugly dimensionality stuff...
        return self.projection.transform(in_vec.reshape(1,-1)).ravel()

def get_projection(observation_space, max_dims):
    in_shape = observation_space.shape
    if isinstance(observation_space, gym.spaces.Box):
        if max_dims >= np.prod(in_shape):
            projection = FlattenProjection(in_shape=in_shape)
        else:
            projection = RandomProjection(in_shape=in_shape, out_dims=max_dims)
    elif isinstance(observation_space, gym.spaces.Discrete): #TODO: ?
        projection = FlattenProjection(in_shape=in_shape)
    else:
        raise RuntimeError('invalid environment: obs space must be Box or Discrete')
    return projection

def get_num_actions(action_space):
    if not isinstance(action_space, gym.spaces.Discrete):
        raise RuntimeError('invalid environment: action space must be Discrete')
    return action_space.n
    
class EpisodicControlAgent(object):
    def __init__(self, action_space, observation_space, k=None, regressor_type=None, eps=None, max_dims=None):
        if k is None:
            k = args.k
        if regressor_type is None:
            regressor_type = args.regressor_type
        if eps is None:
            eps = args.eps
        if max_dims is None:
            max_dims = args.max_dims

        self.obs_projection = get_projection(observation_space, max_dims)
        self.num_actions = get_num_actions(action_space)

        forests = [KForest(dim=self.obs_projection.out_dims(), memory_size=args.memory_size, spill=args.spill, num_trees=args.num_trees) for _ in range(self.num_actions)]
        reg_ctor = LocalConstantReg if args.regressor_type == 'constant' else LocalLinearReg
        self.action_buffers = [reg_ctor(k, forest) for forest in forests]

        self.eps = eps

    def init_episode(self, obs):
        self.history = []

    def observe_action(self, action, reward, obs_pre, obs_post):
        self.history.append((action, reward, self.obs_projection(obs_pre)))

    def wrapup_episode(self):
        return_ = 0
        for action,reward,obs_pre in reversed(self.history):
            self.action_buffers[action].add(obs_pre, return_)
            return_ += reward

    def choose_action(self, obs):
        if args.dry_run:
            return np.random.choice(self.num_actions)

        obs = self.obs_projection(obs)
        estimates = np.array([buff.query(obs)[0] for buff in self.action_buffers])
        s = sum(estimates)
        if random.uniform(0,1) < self.eps or not np.isfinite(s):
            #return random.randint(0,len(self.action_buffers)-1)
            return np.random.choice(len(estimates))
        #return np.random.choice(len(estimates), p=estimates/s)
        return np.argmax(estimates)
