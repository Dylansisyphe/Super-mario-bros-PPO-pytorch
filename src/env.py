"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

# 对图像进行预处理，将图像转换为灰度图像，并将尺寸缩放为84*84
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    # 执行一步游戏
    def step(self, action):
        # 执行游戏
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        # 对图像进行处理
        state = process_frame(state)
        # 计算当前奖励
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        # 游戏结束
        if done:
            # 判断结束对原因
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info
    # 重置游戏状态
    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    # 执行一步游戏
    def step(self, action):
        # 总奖励的分数
        total_reward = 0
        # 多步的游戏状态
        last_states = []
        # 执行多步游戏
        for i in range(self.skip):
            # 执行一步游戏
            state, reward, done, info = self.env.step(action)
            # 记录增加的分数
            total_reward += reward
            # 取这几步游戏的中间帧
            if i >= self.skip / 2:
                last_states.append(state)
            # 如果游戏结束，则重置游戏
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        # 将两帧图片拼接起来
        max_state = np.max(np.concatenate(last_states, 0), 0)
        # 指定前面三帧都是上三个的
        self.states[:-1] = self.states[1:]
        # 最后一帧指定为当前的游戏帧
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    # 重置游戏状态
    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


# 创建游戏环境
def create_train_env(world, stage, actions, output_path=None):
    # 创建游戏
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    # 简化游戏对象
    env = JoypadSpace(env, actions)
    # 自定义奖励逻辑
    env = CustomReward(env, world, stage, monitor)
    # 自定义执行游戏帧
    env = CustomSkipFrame(env)

    return env


# 定义多进程环境
class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        if action_type == "right":
            # 只向右走
            actions = RIGHT_ONLY
        elif action_type == "simple":
            # 7种简单的动作
            actions = SIMPLE_MOVEMENT
        else:
            # 非常复杂的动作，考虑到网络训练的难易程度，不使用这类动作
            actions = COMPLEX_MOVEMENT
        # 创建多个游戏环境
        self.envs = [create_train_env(world, stage, actions, output_path=output_path) for _ in range(num_envs)]
        # 获取单个环境中图像的数量
        self.num_states = self.envs[0].observation_space.shape[0]
        # 获取动作的数量
        self.num_actions = len(actions)
        # 打开多个进程
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            # print(process)
            process.start()
            self.env_conns[index].close()

    # 执行游戏动作
    def run(self, index):
        self.agent_conns[index].close()
        while True:
            # 接收发过来的游戏动作
            request, action = self.env_conns[index].recv()
            if request == "step":
                # 执行一步游戏
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                # 重置游戏状态
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
