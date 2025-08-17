# 导入强化学习环境库
import gymnasium as gym
# 导入绘图库
import matplotlib.pyplot as plt
# 导入时间模块，用于基准测试
import time
# 导入数值计算库
import numpy as np

# 导入PyTorch深度学习框架
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入函数式接口
import torch.nn.functional as F
# 导入优化器
import torch.optim as optim
# 导入数值计算库
import numpy as np
# 导入集合模块，用于双端队列作为记忆缓冲区
import collections
# 导入随机数模块
import random


# 设置设备，优先使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(object):
    """记忆缓冲区类，用于存储经验回放数据"""
    def __init__(self, max_size):
        # 设置记忆缓冲区最大大小
        self.memory_size = max_size
        # 转换计数器，记录记忆中的转换数量
        # 这个计数是必需的，用于延迟学习直到缓冲区合理填满
        self.trans_counter=0
        # 当前缓冲区指针
        self.index=0
        # 使用双端队列作为缓冲区
        self.buffer = collections.deque(maxlen=self.memory_size)
        # 定义转换的命名元组结构
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        """保存一个转换到记忆缓冲区"""
        # 创建转换元组
        t = self.transition(state, action, reward, new_state, terminal)
        # 添加到缓冲区
        self.buffer.append(t)
        # 更新转换计数器
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        """从记忆缓冲区中随机采样一批数据"""
        # 断言缓冲区中有足够的数据进行采样
        assert len(self.buffer) >= batch_size
        # 随机采样指定数量的转换
        transitions = random.sample(self.buffer, k=batch_size)
        # 提取状态并转换为张量
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        # 提取动作并转换为张量
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        # 提取奖励并转换为张量
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        # 提取新状态并转换为张量
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        # 提取终止标志并转换为张量
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        # 返回所有张量
        return states, actions, rewards, new_states, terminals

class QNN(nn.Module):
    """Q网络类，定义深度神经网络结构"""
    def __init__(self, state_size, action_size, seed):
        # 调用父类初始化
        super(QNN, self).__init__()
        # 设置随机种子
        self.seed = torch.manual_seed(seed)
        # 第一层全连接层
        self.fc1 = nn.Linear(state_size, 128)
        # 第二层全连接层
        self.fc2 = nn.Linear(128, 128)
        # 第三层全连接层（输出层）
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        """前向传播"""
        # 第一层全连接
        x = self.fc1(state)
        # ReLU激活函数
        x = F.relu(x)
        # 第二层全连接
        x = self.fc2(x)
        # ReLU激活函数
        x = F.relu(x)
        # 输出层
        return self.fc3(x)

class Agent(object):
    """基础智能体类"""
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, is_learning=True):
        # 折扣因子
        self.gamma = gamma
        # 探索率
        self.epsilon = epsilon
        # epsilon衰减率
        self.epsilon_dec = epsilon_dec
        # epsilon最小值
        self.epsilon_min = epsilon_end
        # 批次大小
        self.batch_size = batch_size
        # 是否在学习模式
        self.is_learning = is_learning
        # 创建记忆缓冲区
        self.memory = MemoryBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        """保存经验到记忆缓冲区"""
        self.memory.save(state, action, reward, new_state, done)  

    def reduce_epsilon(self):
        """减少探索率"""
        # 如果epsilon大于最小值，则按衰减率减少
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  
        
        
        
    
class DoubleQAgent(Agent):
    """双深度Q网络智能体类"""
    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100,
                 is_learning=True):
        
        # 调用父类初始化
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size, is_learning=is_learning)

        # 目标网络更新频率
        self.replace_q_target = replace_q_target
        # 创建主Q网络
        self.q_func = QNN(observation_space_shape, action_space_n, 42).to(device)
        # 创建目标Q网络
        self.q_func_target = QNN(observation_space_shape, action_space_n, 42).to(device)
        # 创建Adam优化器
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)

    
    def choose_action(self, state):
        """选择动作的方法"""
        # 生成随机数
        rand = np.random.random()
        # 将状态转换为张量
        state = torch.from_numpy(state).float().unsqueeze(0)
        # 设置为评估模式
        self.q_func.eval()
        # 不计算梯度
        with torch.no_grad():
            # 获取动作值
            action_values = self.q_func(state)
        # 设置为训练模式
        self.q_func.train()
        # 如果随机数大于epsilon或不在学习模式，选择最优动作
        if rand > self.epsilon or self.is_learning == False: 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # 探索：返回随机动作
            return np.random.choice([i for i in range(4)])   
        
        
    def learn(self):
        """学习方法"""
        # 如果记忆中的转换数量少于批次大小，等待更多数据
        if self.memory.trans_counter < self.batch_size:
            return
            
        # 1. 从过去的转换中选择样本
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. 更新目标值
        # 使用目标网络计算下一状态的Q值
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        # 计算更新的Q值
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        # 获取当前状态的Q值
        q = self.q_func(states).gather(1, actions)
        
        # 3. 更新主神经网络
        # 计算均方误差损失
        loss = F.mse_loss(q, q_updated)
        # 清零梯度
        self.optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        self.optimizer.step()
        
        # 4. 更新目标网络（每N步更新一次）
        if self.memory.trans_counter % self.replace_q_target == 0:
            # 将主网络的参数复制到目标网络
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)
                
        # 5. 减少探索率
        self.reduce_epsilon()