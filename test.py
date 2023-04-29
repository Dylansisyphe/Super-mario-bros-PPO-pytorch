'''
Author: dylansisyphe dylansisyphe@gmail.com
Date: 2023-04-28 21:54:50
LastEditors: dylansisyphe dylansisyphe@gmail.com
LastEditTime: 2023-04-29 12:40:49
FilePath: /Super-mario-bros-PPO-pytorch/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--world", type=int, default=3)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output_self")
    args = parser.parse_args()
    return args


def test(opt):
    # 固定初始化状态，和训练的初始化状态一致
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # 判断游戏动作类型
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    # 创建游戏环境
    env = create_train_env(opt.world, opt.stage, actions,
                           "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
    # 创建模型
    model = PPO(env.observation_space.shape[0], len(actions))
    # 加载模型参数文件
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                         map_location=lambda storage, loc: storage))
    # 切换评估模式
    model.eval()
    # 获取刚开始的游戏图像
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        # 显示界面
        env.render()
        # 预测动作概率和评估值
        logits, value = model(state)
        # 获取动作的序号
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        # 执行游戏
        state, reward, done, info = env.step(action)
        # 转换每一步都游戏状态
        state = torch.from_numpy(state)
        # 游戏通关
        if info["flag_get"]:
            print("World {} stage {} completed".format(opt.world, opt.stage))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
