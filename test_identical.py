import sys, random, threading, time, gzip, traceback
import gym, gym_ple
import numpy as np
import os
from args import *
from utils import StopWatch, timeF
from Queue import Queue

arg_parser.add_argument('env')
arg_parser.add_argument('seed', type=int, default=1)

class FakeAgent(object):
    def choose_action(*args, **kwargs):
        return 0

class AgentProcess(object):
    def __init__(self):
        self.num_repeat = 6
        self.env = gym.make(args.env)
        self.agent = FakeAgent()

    def act_episode(self):
        self.env.seed(env.seed)

        obs = self.env.reset()

        frames = 0
        return_ = 0
        do_render = os.path.isfile('/tmp/gymrender')

        while True:
            action = 1
            reward = 0
            for _ in range(self.num_repeat):
                obs, sub_reward, done, info = self.env.step(action)
                if do_render:
                    self.env.render()
                frames += 1
                reward += sub_reward
                if done:
                    break

            return_ += reward
            if done:
                return (return_, frames)

def log_episode(*args):
    sys.stdout.write(','.join(map(str, args))+'\n')
    sys.stdout.flush()

if __name__ == '__main__':
    parse_args()
    
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    agent_process = AgentProcess()

    for episode_num in range(100):
        return_, ep_frames = agent_process.act_episode()
        log_episode(episode_num, return_, ep_frames)
