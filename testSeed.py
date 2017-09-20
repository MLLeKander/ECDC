import gym, gym_ple, time, os
from args import *
from EC_agent import RandomProjection
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ['SDL_VIDEODRIVER'] = 'dummy'

arg_parser.add_argument('--frameskip', type=int)
arg_parser.add_argument('--steps', type=int)
arg_parser.add_argument('--l', type=int)
arg_parser.add_argument('--env', default='MsPacmanNoFrameskip-v4')
parse_args()


env = gym.make(args.env)
projection = RandomProjection(in_shape=env.observation_space.shape, out_dims=32)
#projection = lambda x: x
def run_for(i):
    hist, rewards = [], []

    env = gym.make(args.env)

    init_frame = env.reset()
    hist.append(init_frame)
    action = 0
    for i in range(args.steps):
        reward = 0
        for _ in range(args.frameskip):
            frame, sub_reward, done, _ = env.step(action%env.action_space.n)
            action += 1
            reward += sub_reward
            if done: break
        frame_proj = projection(frame)
        hist.append(frame_proj)
        rewards.append(reward)
        if done: break
    print len(hist), env.env.ale.getInt(b'random_seed')
    return hist, rewards

def match(run1, run2):
    if len(run1) != len(run2):
        return False
    for frame1, frame2 in zip(run1[0], run2[0]):
        if not (frame1 == frame2).all():
            return False
    for reward1, reward2 in zip(run1[1], run2[1]):
        if reward1 != reward2:
            return False
    return True

hists = [run_for(i) for i in range(args.l)]
#hists.extend([run_for(i+123) for i in range(args.l)])
print map(len, [h[0] for h in hists])

for ndx1 in range(0, len(hists)):
    for ndx2 in range(ndx1+1, len(hists)):
        if ndx1 == ndx2%args.l:
            if not match(hists[ndx1], hists[ndx2]):
                print '%d != %d'%(ndx1, ndx2)
        else:
            if match(hists[ndx1], hists[ndx2]):
                print '%d == %d'%(ndx1, ndx2)
