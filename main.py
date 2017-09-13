import sys, random, threading, time, gzip, traceback
import gym, gym_ple
import numpy as np
import os
from args import *
from utils import StopWatch
from Queue import Queue
from EC_agent import EpisodicControlAgent

arg_parser.add_argument('--max_episodes', type=int, default=1000000)
arg_parser.add_argument('--max_frames', type=int, default=5000000)
arg_parser.add_argument('--num_repeat', type=int, default=4)
arg_parser.add_argument('--force_overwrite', type=str2bool, default=False)
arg_parser.add_argument('--headless', type=str2bool, default=False)
arg_parser.add_argument('env')
arg_parser.add_argument('log_dir')

class AgentProcess(object):
    def __init__(self, env_name, seed, num_repeat):
        self.seed = seed
        self.num_repeat = num_repeat

        self.act_timer = StopWatch()
        self.wrapup_timer = StopWatch()

        self.env = gym.make(env_name)
        self.agent = EpisodicControlAgent(self.env.action_space, self.env.observation_space)

        np.random.seed(self.seed)
        self.env._seed(self.seed)
        random.seed(self.seed)

    def run_episode(self):
        self.act_timer.start()
        return_, frames = self.act_episode()
        self.act_timer.pause()

        self.wrapup_timer.start()
        self.wrapup_episode()
        self.wrapup_timer.pause()
        return return_, frames, self.act_timer.time(), self.wrapup_timer.time()

    def act_episode(self):
        obs = self.env.reset()
        self.agent.init_episode(obs)

        frames = 0
        return_ = 0
        do_render = os.path.isfile('/tmp/gymrender')

        while True:
            action = self.agent.choose_action(obs)
            obs_pre = obs

            reward = 0
            for _ in range(self.num_repeat):
                obs, sub_reward, done, info = self.env.step(action)
                if do_render:
                    self.env.render()
                frames += 1
                reward += sub_reward
                if done:
                    break

            self.agent.observe_action(action, reward, obs_pre, obs)
            return_ += reward
            if done:
                return (return_, frames)

    def wrapup_episode(self):
        self.agent.wrapup_episode()

def ensure_log_dir():
    if os.path.exists(args.log_dir):
        if not args.force_overwrite:
            raise ValueError('log_dir exists (and --force_overwrite as False)')
    else:
        os.makedirs(args.log_dir)

def write_arg_file():
    arg_file = open(os.path.join(args.log_dir, 'args'), 'w')
    arg_file.write(`vars(args)`)
    arg_file.close()

def get_run_file():
    return open(os.path.join(args.log_dir, 'run.csv'), 'w')

def get_buff_files(n):
    return data_files, label_files

def write_buff_files(agent_process):
    def write_forest_to(forest, data_file, label_file):
        for i in range(forest.get_memory_size()):
            if forest.is_active(i):
                data_file.write(' '.join('%.3f'%dat for dat in forest.get_data(i)) + '\n')
                label_file.write('%.3f\n'%forest.get_label(i))

    for ndx, buf in enumerate(agent_process.agent.action_buffers):
        data_fname = os.path.join(args.log_dir, 'data_%d.gz' % ndx)
        label_fname = os.path.join(args.log_dir, 'labels_%d.gz' % ndx)
        with gzip.open(data_fname, 'wb') as data_file, gzip.open(label_fname, 'wb') as label_file:
            write_forest_to(buf.forest, data_file, label_file)

if __name__ == '__main__':
    parse_args()
    
    if args.headless:
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env_name = args.env

    agent_process = AgentProcess(env_name=env_name, seed=5, num_repeat=args.num_repeat)

    ensure_log_dir()
    write_arg_file()
    run_file = get_run_file()

    run_file.write('episode,totalBufferSize,totalFrameCount,walltime,return,epFrames,actTime,wrapupTime')
    for i in range(agent_process.env.action_space.n):
        run_file.write(',size%d'%i)
    run_file.write('\n')

    stopwatch = StopWatch()
    stopwatch.start()

    total_frame_count = 0
    try:
        for episode in range(args.max_episodes):
            ep_data = agent_process.run_episode()

            buff_sizes = [buff.size() for buff in agent_process.agent.action_buffers]
            total_buffer_size = sum(buff_sizes)
            total_frame_count += ep_data[1]

            outputs = [episode, total_buffer_size, total_frame_count, stopwatch.time()]
            outputs.extend(ep_data)
            outputs.extend(buff_sizes)

            run_file.write(','.join(map(str, outputs))+'\n')
            run_file.flush()
            if total_frame_count >= args.max_frames:
                break
    except:
        traceback.print_exc()
    finally:
        run_file.close()
        write_buff_files(agent_process)

#episode, return, totalBufferSize, totalFrameCount, walltime, epFrames, actTimer, wrapupTimer
# per-buffer: size
