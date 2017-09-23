import gym
import random
import numpy as np
import cv2
from args import *
from localreg import LocalConstantReg, LocalLinearReg
from vqtree import KForest
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

arg_parser.add_argument('--num_trees', type=int, default=1)
arg_parser.add_argument('--memory_size', type=int, default=1000000)
arg_parser.add_argument('--max_leaf_size', type=int, default=128)
arg_parser.add_argument('--branch_factor', type=int, default=32)
arg_parser.add_argument('--spill', type=float, default=-1)
arg_parser.add_argument('--min_leaves', type=int, default=40)
arg_parser.add_argument('--search_type', type=int, default=3)
arg_parser.add_argument('--exact_eps', type=float, default=0.1)

arg_parser.add_argument('--project_gauss', type=str2bool, default=False)
arg_parser.add_argument('--rescale_height', type=int, default=-1)
arg_parser.add_argument('--rescale_width', type=int, default=-1)

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
    def __init__(self, in_shape, out_dims, seed):
        if args.project_gauss:
            self.projection = GaussianRandomProjection(n_components=out_dims, random_state=seed)
        else:
            self.projection = SparseRandomProjection(n_components=out_dims, random_state=seed)
        num_dims = np.prod(in_shape)
        self.projection.fit([[1]*num_dims])
        self._out_dims = out_dims

    def out_dims(self):
        return self._out_dims

    def __call__(self, in_vec):
        # Ugly dimensionality stuff...
        return self.projection.transform(in_vec.reshape(1,-1)).ravel()

class RescaleProjection(object):
    def __init__(self, in_shape, out_dims, seed):
        rand_in_shape = list(in_shape) # Temporarily convert to list to modify
        rand_in_shape[0] = args.rescale_width
        rand_in_shape[1] = args.rescale_height
        rand_in_shape = tuple(rand_in_shape)

        self.rand_projection = RandomProjection(rand_in_shape, out_dims, seed)

    def out_dims(self):
        return self.rand_projection.out_dims()

    def __call__(self, in_vec):
        rescaled = cv2.resize(in_vec, (args.rescale_width, args.rescale_height), interpolation=cv2.INTER_LINEAR)
        return self.rand_projection(rescaled)

def make_buffers(env_name, k=None, regressor_type=None, max_dims=None, seed=5):
    def get_projection(observation_space, max_dims):
        in_shape = observation_space.shape
        if isinstance(observation_space, gym.spaces.Box):
            if max_dims >= np.prod(in_shape):
                projection = FlattenProjection(in_shape=in_shape)
            else:
                if (args.rescale_height <= 0) != (args.rescale_width <= 0):
                    raise ValueError('Either both or neither of rescale_height and rescale_width must be specified')
                elif args.rescale_height > 0:
                    projection = RescaleProjection(in_shape=in_shape, out_dims=max_dims, seed=seed)
                else:
                    projection = RandomProjection(in_shape=in_shape, out_dims=max_dims, seed=seed)
        elif isinstance(observation_space, gym.spaces.Discrete): #TODO: ?
            projection = FlattenProjection(in_shape=in_shape)
        else:
            raise RuntimeError('invalid environment: obs space must be Box or Discrete')
        return projection

    env = gym.make(env_name)
    action_space, observation_space = env.action_space, env.observation_space

    if k is None:
        k = args.k
    if regressor_type is None:
        regressor_type = args.regressor_type
    if max_dims is None:
        max_dims = args.max_dims

    obs_projection = get_projection(observation_space, max_dims)

    if not isinstance(action_space, gym.spaces.Discrete):
        raise RuntimeError('invalid environment: action space must be Discrete')
    num_actions = action_space.n

    forest_arg_names = ['memory_size', 'max_leaf_size', 'branch_factor', 'spill', 'num_trees', 'min_leaves', 'exact_eps', 'search_type']
    forest_args = {name:vars(args)[name] for name in forest_arg_names}
    forests = [KForest(dim=obs_projection.out_dims(), rand_seed=seed, remove_dups=True, **forest_args) for _ in range(num_actions)]

    env.close()

    reg_ctor = LocalConstantReg if args.regressor_type in ['constant', 'kernel'] else LocalLinearReg
    return [reg_ctor(k, forest) for forest in forests], obs_projection
    
class EpisodicControlAgent(object):
    def __init__(self, action_buffers, obs_projection, eps=None):
        if eps is None:
            eps = args.eps

        self.action_buffers = action_buffers
        self.obs_projection = obs_projection
        self.num_actions = len(action_buffers)
        self.eps = eps

    def init_episode(self, obs):
        self.history = []

    def observe_action(self, action, reward, obs_pre, obs_post, meta):
        self.history.append((action, reward, self.obs_projection(obs_pre), meta))

    def wrapup_episode(self):
        return_ = 0
        for action,reward,obs_pre,meta in reversed(self.history):
            return_ += reward
            self.action_buffers[action].update_drift(obs_pre, return_, meta)

        for action_buffer in self.action_buffers:
            action_buffer.enforce_drift()

        return_ = 0
        for action,reward,obs_pre,meta in reversed(self.history):
            return_ += reward
            self.action_buffers[action].add(obs_pre, return_)

    def choose_action(self, obs):
        if args.dry_run:
            return np.random.choice(self.num_actions)

        obs = self.obs_projection(obs)
        queries = [buff.query(obs) for buff in self.action_buffers]
        estimates = np.array([query[0] for query in queries])

        s = sum(estimates)
        if random.uniform(0,1) < self.eps or not np.isfinite(s):
            return np.random.choice(len(estimates)), None

        #action = np.random.choice(len(estimates), p=estimates/s)
        action = np.argmax(estimates)
        return action, queries[action][1]
