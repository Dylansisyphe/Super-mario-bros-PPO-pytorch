"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

# 强化学习玩超级马里奥的思路是观察当前帧图片，根据每一帧图像，输出要执行的action

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    # 固定初始化状态
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 创建保存模型的文件夹， 若文件夹已存在，递归删除文件夹以及里面的文件
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    # 为游戏评估单独开一个进程
    # mp = _mp.get_context("spawn")
    # 创建多进程的游戏环境
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory() # 允许多个进程共享一块内存,由此来达到交换信息的进程通信机制;这里指多个进程间共用一套模型参数
    # process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    # process.start()
    # 创建优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # 刚开始给每个进程的游戏执行初始化
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    # 获取游戏初始的界面
    for agent_conn in envs.agent_conns:
        print(agent_conn)
        agent_conn.recv()
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    # print('curr_states: ' + curr_states)
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0
    while True:
        # if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            # 执行预测
            logits, value = model(curr_states)
            values.append(value.squeeze())
            # 计算每个动作的概率值
            policy = F.softmax(logits, dim=1)
            # 根据每个标签的概率随机生成符合概率的标签
            old_m = Categorical(policy)
            action = old_m.sample()
            # 记录预测数据
            actions.append(action)
            # 计算类别的概率的对数
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            # 向各个进程游戏发送动作
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
            # 将多进程的游戏数据打包
            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            print("info:"+info)
            # 进行数据转换
            state = torch.from_numpy(np.concatenate(state, 0))
            # 转换为tensor数据
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            # 记录预测数据
            rewards.append(reward)
            dones.append(done)
            curr_states = state
        # 根据上面最后的图像预测
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)

        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                # 根据拿到的图像执行预测
                logits, value = model(states[batch_indices])
                # 计算每个动作的概率值
                new_policy = F.softmax(logits, dim=1)
                # 计算类别的概率的对数
                new_m = Categorical(new_policy) # 生成类别分布
                new_log_policy = new_m.log_prob(actions[batch_indices])
                 # 计算actor损失
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[
                                                       batch_indices]))
                # 计算critic损失
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                # 计算全部损失
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                # 计算梯度
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
