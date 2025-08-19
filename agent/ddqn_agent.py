import gymnasium as gym
import matplotlib.pyplot as plt
import time # for benchmarking
import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections # For dequeue for the memory buffer
import random


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
if torch.cuda.is_available():
    print(f"🔧 CUDA版本: {torch.version.cuda}")
    print(f"🎮 GPU数量: {torch.cuda.device_count()}")
    print(f"📱 当前GPU: {torch.cuda.get_device_name(0)}")

def ensure_tensor_on_device(tensor, target_device):
    """确保张量在目标设备上"""
    if tensor.device != target_device:
        return tensor.to(target_device)
    return tensor

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter=0 # num of transitions in the memory
                             # this count is required to delay learning
                             # until the buffer is sensibly full
        self.index=0         # current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, new_states, terminals

class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, is_learning=True):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.is_learning = is_learning
        self.memory = MemoryBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)  

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  
        
        
        
    
class DoubleQAgent(Agent):
    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100,
                 is_learning=True):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size, is_learning=is_learning)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(observation_space_shape, action_space_n, 42).to(device)
        self.q_func_target = QNN(observation_space_shape, action_space_n, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)

    
    def choose_action(self, state):
        rand = np.random.random()
        
        # 确保输入状态是numpy数组
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # 转换为torch张量并移动到正确设备
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # 验证设备一致性
        if state_tensor.device != device:
            print(f"⚠️  警告: 状态张量设备不匹配，从 {state_tensor.device} 移动到 {device}")
            state_tensor = state_tensor.to(device)
        
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state_tensor)
        self.q_func.train()
        
        if rand > self.epsilon or self.is_learning == False: 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(4)])   
        
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 确保所有张量都在正确设备上
        states = ensure_tensor_on_device(states, device)
        actions = ensure_tensor_on_device(actions, device)
        rewards = ensure_tensor_on_device(rewards, device)
        new_states = ensure_tensor_on_device(new_states, device)
        terminals = ensure_tensor_on_device(terminals, device)
        
        # 2. Update the target values
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main NN
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Update the target NN (every N-th step)
        if self.memory.trans_counter % self.replace_q_target == 0: # wait before you start learning
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)
                
        # 5. Reduce the exploration rate
        self.reduce_epsilon()
    
    def save_model(self, filepath, save_optimizer=True, save_memory=False):
        """保存训练好的模型
        
        Args:
            filepath (str): 模型保存路径
            save_optimizer (bool): 是否保存优化器状态
            save_memory (bool): 是否保存经验回放缓冲区
        """
        import os
        import json
        
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型状态
        model_state = {
            'q_func_state_dict': self.q_func.state_dict(),
            'q_func_target_state_dict': self.q_func_target.state_dict(),
            'observation_space_shape': self.q_func.fc1.in_features,
            'action_space_n': self.q_func.fc3.out_features,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_dec': self.epsilon_dec,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'lr': self.optimizer.param_groups[0]['lr'],
            'replace_q_target': self.replace_q_target,
            'mem_size': self.memory.memory_size,
            'device': str(device)
        }
        
        # 保存优化器状态
        if save_optimizer:
            model_state['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # 保存经验回放缓冲区
        if save_memory:
            model_state['memory_buffer'] = {
                'trans_counter': self.memory.trans_counter,
                'buffer_size': len(self.memory.buffer),
                'transitions': list(self.memory.buffer)
            }
        
        # 保存模型
        torch.save(model_state, filepath)
        
        # 保存模型信息
        info_filepath = filepath.replace('.pth', '_info.json')
        with open(info_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': 'DDQN',
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': {
                    'observation_space': model_state['observation_space_shape'],
                    'action_space': model_state['action_space_n'],
                    'gamma': model_state['gamma'],
                    'epsilon': model_state['epsilon'],
                    'learning_rate': model_state['lr']
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 模型已保存到: {filepath}")
        print(f"📋 模型信息已保存到: {info_filepath}")
    
    def load_model(self, filepath, load_optimizer=True, load_memory=False):
        """加载训练好的模型
        
        Args:
            filepath (str): 模型文件路径
            load_optimizer (bool): 是否加载优化器状态
            load_memory (bool): 是否加载经验回放缓冲区
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        print(f"🔄 正在加载模型: {filepath}")
        
        # 加载模型状态
        checkpoint = torch.load(filepath, map_location=device)
        
        # 验证模型兼容性
        if checkpoint['observation_space_shape'] != self.q_func.fc1.in_features:
            raise ValueError(f"观察空间不匹配: 期望 {self.q_func.fc1.in_features}, 得到 {checkpoint['observation_space_shape']}")
        
        if checkpoint['action_space_n'] != self.q_func.fc3.out_features:
            raise ValueError(f"动作空间不匹配: 期望 {self.q_func.fc3.out_features}, 得到 {checkpoint['action_space_n']}")
        
        # 加载模型权重
        self.q_func.load_state_dict(checkpoint['q_func_state_dict'])
        self.q_func_target.load_state_dict(checkpoint['q_func_target_state_dict'])
        
        # 加载优化器状态
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ 优化器状态已加载")
        
        # 加载经验回放缓冲区
        if load_memory and 'memory_buffer' in checkpoint:
            self.memory.trans_counter = checkpoint['memory_buffer']['trans_counter']
            self.memory.buffer = collections.deque(checkpoint['memory_buffer']['transitions'], 
                                                 maxlen=self.memory.memory_size)
            print("✅ 经验回放缓冲区已加载")
        
        # 更新其他参数
        self.gamma = checkpoint['gamma']
        self.epsilon = checkpoint['epsilon']
        self.epsilon_dec = checkpoint['epsilon_dec']
        self.epsilon_min = checkpoint['epsilon_min']
        self.batch_size = checkpoint['batch_size']
        self.replace_q_target = checkpoint['replace_q_target']
        
        print("✅ 模型加载完成!")
        print(f"📊 模型信息:")
        print(f"   - 观察空间: {checkpoint['observation_space_shape']}")
        print(f"   - 动作空间: {checkpoint['action_space_n']}")
        print(f"   - 当前探索率: {self.epsilon:.4f}")
        print(f"   - 折扣因子: {self.gamma}")
        print(f"   - 学习率: {checkpoint['lr']}")
    
    def export_onnx(self, filepath, input_shape=(1, 7)):
        """导出模型为ONNX格式，用于生产部署
        
        Args:
            filepath (str): ONNX文件保存路径
            input_shape (tuple): 输入张量形状
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            print("❌ 需要安装onnx和onnxruntime: pip install onnx onnxruntime")
            return False
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape).to(device)
        
        # 设置为评估模式
        self.q_func.eval()
        
        # 导出ONNX
        torch.onnx.export(
            self.q_func,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX模型已导出到: {filepath}")
        print(f"📊 模型输入形状: {input_shape}")
        print(f"🔍 ONNX模型验证通过")
        
        return True
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'DDQN',
            'observation_space': self.q_func.fc1.in_features,
            'action_space': self.q_func.fc3.out_features,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'device': str(device),
            'total_parameters': sum(p.numel() for p in self.q_func.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.q_func.parameters() if p.requires_grad)
        }