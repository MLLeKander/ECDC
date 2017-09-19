import sys, random, threading, time, gzip, traceback
import gym, gym_ple
import numpy as np
import os
from args import *
from utils import StopWatch, timeF
from Queue import Queue
from EC_agent import EpisodicControlAgent, make_buffers

arg_parser.add_argument('--max_episodes', type=int, default=1000000)
arg_parser.add_argument('--max_frames', type=int, default=5000000)
arg_parser.add_argument('--num_repeat', type=int, default=1)
arg_parser.add_argument('--force_overwrite', type=str2bool, default=False)
arg_parser.add_argument('--headless', type=str2bool, default=False)
arg_parser.add_argument('--checkpoint_frame_spacing', type=int, default=-1)
arg_parser.add_argument('--eval_frame_spacing', type=int, default=-1)
arg_parser.add_argument('--eval_seeds', type=str2list(int), default=range(100,105))
arg_parser.add_argument('--multi_eval', type=str2bool, default=False)
arg_parser.add_argument('--seed', type=int, default=5)
arg_parser.add_argument('--atari_greyscale','--atari_grayscale', type=str2bool, default=True)
arg_parser.add_argument('env')
arg_parser.add_argument('log_dir')

class AgentProcess(object):
    def __init__(self, env_name, buffers, projection, seed, num_repeat):
        self.seed = seed
        self.num_repeat = num_repeat

        self.frame_count = 0

        self.env = gym.make(env_name)
        self.agent = EpisodicControlAgent(buffers, projection)

        np.random.seed(self.seed)
        self.env.seed(self.seed)
        random.seed(self.seed)

    def run_episode(self):
        act_time, (return_, frames) = timeF(self.act_episode)
        wrapup_time, _ = timeF(self.wrapup_episode)
        self.frame_count += frames

        return return_, frames, self.frame_count, act_time, wrapup_time

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

def get_train_file():
    return open(os.path.join(args.log_dir, 'train.csv'), 'w')

def get_eval_file():
    return open(os.path.join(args.log_dir, 'eval.csv'), 'w')

def write_buff_files(agent_process):
    def get_forest_data(forest):
        datas, labels = [], []
        for i in range(forest.get_memory_size()):
            if forest.is_active(i):
                datas.append(forest.get_data(i))
                labels.append(forest.get_label(i))
        return datas, labels

    all_data = {}
    for ndx, buf in enumerate(agent_process.agent.action_buffers):
        datas, labels = get_forest_data(buf.forest)
        all_data['data_%d'%ndx] = datas
        all_data['labels_%d'%ndx] = labels
    fname = os.path.join(args.log_dir, 'data.npz')
    np.savez_compressed(fname, **all_data)

def log_init(log_file, buffers):
    log_file.write('episode,totalBufferSize,totalFrameCount,walltime,return,epFrames,actTime,wrapupTime')
    for i in range(len(buffers)):
        log_file.write(',size%d'%i)
    log_file.write(',seed\n')
    log_file.flush()

def log_episode(log_file, buffers, episode_num, time, return_, ep_frames, total_frame_count, act_time, wrapup_time, seed):
    buff_sizes = [buff.size() for buff in buffers]
    total_buffer_size = sum(buff_sizes)

    outputs = [episode_num, total_buffer_size, total_frame_count, time]
    outputs.extend([return_, ep_frames, act_time, wrapup_time])
    outputs.extend(buff_sizes)
    outputs.append(seed)

    log_file.write(','.join(map(str, outputs))+'\n')
    log_file.flush()

#TODO: This is really ugly monkeypatching
# A proper solution would be to properly subclass, but this gets the job done for now...
def monkeypatch_atari_greyscale():
    from gym.envs.atari import AtariEnv

    AtariEnv._get_image = lambda slf: slf.ale.getScreenGrayscale().squeeze()

    old_init = AtariEnv.__init__
    def new_init(slf, *args, **kwargs):
        old_init(slf, *args, **kwargs)
        old_space = slf.observation_space
        if len(old_space.shape) == 3:
            slf.observation_space = gym.spaces.Box(low=old_space.low[:,:,0], high=old_space.high[:,:,0])
    AtariEnv.__init__ = new_init

    old_render = AtariEnv._render
    def new_render(slf, mode='human', close=False):
        if close or mode != 'human':
            old_render(slf, mode, close)
        if slf.viewer is None:
            from gym.envs.classic_control import rendering
            slf.viewer = rendering.SimpleImageViewer()
        slf.viewer.imshow(slf._get_image()[:,:,np.newaxis].repeat(3,axis=2))
    AtariEnv._render = new_render

if __name__ == '__main__':
    parse_args()
    
    if args.headless: # Fix for gym_ple, which likes to needlessly spawn windows on construction...
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    if args.atari_greyscale:
        monkeypatch_atari_greyscale()

    env_name = args.env

    buffers, projection = make_buffers(env_name=env_name, seed=args.seed)
    agent_process = AgentProcess(env_name=env_name, buffers=buffers, projection=projection, seed=args.seed, num_repeat=args.num_repeat)
    eval_process = AgentProcess(env_name=env_name, buffers=buffers, projection=projection, seed=args.seed, num_repeat=args.num_repeat)

    ensure_log_dir()
    write_arg_file()
    train_file, eval_file = get_train_file(), get_eval_file()

    log_init(train_file, buffers)
    log_init(eval_file, buffers)

    next_checkpoint, next_eval = args.checkpoint_frame_spacing, args.eval_frame_spacing
    eval_num = 0

    stopwatch = StopWatch()
    stopwatch.start()
    try:
        for episode_num in range(args.max_episodes):
            return_, ep_frames, total_frame_count, act_time, wrapup_time = agent_process.run_episode()
            log_episode(train_file, buffers, episode_num, stopwatch.time(), return_, ep_frames, total_frame_count, act_time, wrapup_time, args.seed)

            if total_frame_count >= args.max_frames:
                break

            if args.checkpoint_frame_spacing > 0 and total_frame_count >= next_checkpoint:
                stopwatch.pause()

                write_buff_files(agent_process)
                while next_checkpoint < total_frame_count:
                    next_checkpoint += args.checkpoint_frame_spacing

                stopwatch.start()

            if args.eval_frame_spacing > 0 and total_frame_count >= next_eval:
                stopwatch.pause()

                eval_num += 1
                for seed in args.eval_seeds:
                    wrapup_time = 0

                    eval_process.env.seed(seed)
                    act_time, (return_, ep_frames) = timeF(agent_process.act_episode)
                    log_episode(eval_file, buffers, eval_num, stopwatch.time(), return_, ep_frames, total_frame_count, act_time, wrapup_time, seed)

                    if args.multi_eval:
                        eval_process.env.seed(seed)
                        old_eps = eval_process.agent.eps
                        eval_process.agent.eps = -1
                        act_time, (return_, ep_frames) = timeF(agent_process.act_episode)
                        log_episode(eval_file, buffers, eval_num, stopwatch.time(), return_, ep_frames, total_frame_count, act_time, wrapup_time, -seed)

                        eval_process.agent.eps = old_eps
                while next_eval < total_frame_count:
                    next_eval += args.eval_frame_spacing

                stopwatch.start()
    except:
        traceback.print_exc()
    finally:
        stopwatch.pause()

        #eval_num += 1
        #total_frame_count = agent_process.frame_count
        #for seed in args.eval_seeds:
        #    eval_process.env.seed(seed)
        #    act_time, (return_, ep_frames) = timeF(agent_process.act_episode)
        #    wrapup_time = 0
        #    log_episode(eval_file, buffers, eval_num, stopwatch.time(), return_, ep_frames, total_frame_count, act_time, wrapup_time, seed)

        train_file.close()
        eval_file.close()

        write_buff_files(agent_process)

#episodeNum, return, totalBufferSize, totalFrameCount, walltime, epFrames, actTimer, wrapupTimer
# per-buffer: size
