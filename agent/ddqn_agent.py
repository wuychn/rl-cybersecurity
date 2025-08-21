import gymnasium as gym  # 导入强化学习环境库，用于创建和管理训练环境
import matplotlib.pyplot as plt  # 导入绘图库，用于可视化训练过程和结果
import time # for benchmarking  # 导入时间模块，用于性能基准测试
import numpy as np  # 导入数值计算库，用于数组操作和数学计算
import os  # 导入操作系统接口模块，用于文件路径操作
import json  # 导入JSON数据处理模块，用于保存和加载模型信息

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口，提供激活函数等
import torch.optim as optim  # 导入PyTorch优化器模块
import collections # For dequeue for the memory buffer  # 导入集合模块，用于经验回放缓冲区的双端队列
import random  # 导入随机数生成模块，用于探索策略


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 设置计算设备，优先使用GPU 3，否则使用CPU
print(f"🚀 使用设备: {device}")  # 打印当前使用的计算设备
if torch.cuda.is_available():  # 如果CUDA可用
    print(f"🔧 CUDA版本: {torch.version.cuda}")  # 打印CUDA版本信息
    print(f"🎮 GPU数量: {torch.cuda.device_count()}")  # 打印可用的GPU数量
    print(f"📱 当前GPU: {torch.cuda.get_device_name(0)}")  # 打印当前GPU的名称

def ensure_tensor_on_device(tensor, target_device):  # 定义函数，确保张量在目标设备上
    """确保张量在目标设备上 - 如果张量不在目标设备上则移动过去"""
    if tensor.device != target_device:  # 如果张量设备与目标设备不同
        return tensor.to(target_device)  # 将张量移动到目标设备
    return tensor  # 如果已经在目标设备上，直接返回

class MemoryBuffer(object):  # 定义经验回放缓冲区类
    def __init__(self, max_size):  # 初始化方法，接收最大缓冲区大小参数
        self.memory_size = max_size  # 存储最大缓冲区大小
        self.trans_counter=0 # num of transitions in the memory  # 转换计数器，记录缓冲区中的转换数量
                             # this count is required to delay learning  # 这个计数用于延迟学习
                             # until the buffer is sensibly full  # 直到缓冲区合理填满
        self.index=0         # current pointer in the buffer  # 当前缓冲区指针位置
        self.buffer = collections.deque(maxlen=self.memory_size)  # 创建双端队列作为缓冲区，自动限制最大长度
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])  # 定义转换数据结构，包含状态、动作、奖励、新状态和终止标志

    
    def save(self, state, action, reward, new_state, terminal):  # 保存经验到缓冲区的方法
        t = self.transition(state, action, reward, new_state, terminal)  # 创建经验对象
        self.buffer.append(t)  # 将经验添加到缓冲区
        self.trans_counter = (self.trans_counter + 1) % self.memory_size  # 更新经验计数器，使用模运算保持循环

    def random_sample(self, batch_size):  # 随机采样方法，用于经验回放
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full  # 断言缓冲区足够满才开始采样
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample  # 随机采样指定数量的转换
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)  # 提取状态并转换为张量
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)  # 提取动作并转换为张量
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)  # 提取奖励并转换为张量
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)  # 提取新状态并转换为张量
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)  # 提取终止标志并转换为张量
  
        return states, actions, rewards, new_states, terminals  # 返回所有转换数据

class QNN(nn.Module):  # 定义Q网络类，继承自PyTorch的nn.Module
    def __init__(self, state_size, action_size, seed):  # 初始化方法，接收状态大小、动作大小和随机种子
        super(QNN, self).__init__()  # 调用父类初始化方法
        self.seed = torch.manual_seed(seed)  # 设置随机种子，确保结果可重现
        self.fc1 = nn.Linear(state_size, 128)  # 第一个全连接层，从状态大小到128个神经元
        self.fc2 = nn.Linear(128, 128)  # 第二个全连接层，从128到128个神经元
        self.fc3 = nn.Linear(128, action_size)  # 第三个全连接层，从128到动作大小
        
    def forward(self, state):  # 前向传播方法
        x = self.fc1(state)  # 第一层线性变换
        x = F.relu(x)  # 应用ReLU激活函数
        x = self.fc2(x)  # 第二层线性变换
        x = F.relu(x)  # 应用ReLU激活函数
        return self.fc3(x)  # 第三层线性变换，输出Q值

class Agent(object):  # 定义基础智能体类
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, is_learning=True):  # 初始化方法，设置各种超参数
        self.gamma = gamma # alpha = learn rate, gamma = discount  # 折扣因子，用于计算未来奖励的权重
        self.epsilon = epsilon  # 探索率，控制随机探索的概率
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces  # 探索率衰减因子
        self.epsilon_min = epsilon_end  # 最小探索率
        self.batch_size = batch_size  # 批次大小，用于批量学习
        self.is_learning = is_learning  # 是否处于学习模式
        self.memory = MemoryBuffer(mem_size)  # 创建经验回放缓冲区

    def save(self, state, action, reward, new_state, done):  # 保存经验到缓冲区的方法
        self.memory.save(state, action, reward, new_state, done)  # 调用缓冲区的保存方法

    def reduce_epsilon(self):  # 减少探索率的方法
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  # 如果当前探索率大于最小值则衰减，否则保持最小值
        
        
        
    
class DoubleQAgent(Agent):  # 定义双Q网络智能体类，继承自基础智能体
    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100,
                 is_learning=True):  # 初始化方法，设置双Q网络的参数
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size, is_learning=is_learning)  # 调用父类初始化方法

        self.replace_q_target = replace_q_target  # 目标网络更新频率
        self.q_func = QNN(observation_space_shape, action_space_n, 42).to(device)  # 创建主Q网络并移动到指定设备
        self.q_func_target = QNN(observation_space_shape, action_space_n, 42).to(device)  # 创建目标Q网络并移动到指定设备
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)  # 创建Adam优化器

    
    def choose_action(self, state):  # 选择动作的方法
        rand = np.random.random()  # 生成随机数
        
        # 确保输入状态是numpy数组
        if isinstance(state, torch.Tensor):  # 如果状态是PyTorch张量
            state = state.cpu().numpy()  # 转换为numpy数组
        
        # 转换为torch张量并移动到正确设备
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 转换为张量并添加批次维度
        
        # 验证设备一致性
        if state_tensor.device != device:  # 如果张量设备不匹配
            print(f"⚠️  警告: 状态张量设备不匹配，从 {state_tensor.device} 移动到 {device}")  # 打印警告信息
            state_tensor = state_tensor.to(device)  # 移动到正确设备
        
        self.q_func.eval()  # 设置为评估模式
        with torch.no_grad():  # 关闭梯度计算
            action_values = self.q_func(state_tensor)  # 前向传播得到Q值
        self.q_func.train()  # 恢复训练模式
        
        if rand > self.epsilon or self.is_learning == False:  # 如果随机数大于探索率或不在学习模式
            return np.argmax(action_values.cpu().data.numpy())  # 选择Q值最大的动作（贪婪策略）
        else:  # 否则进行探索
            # exploring: return a random action  # 探索：返回随机动作
            return np.random.choice([i for i in range(4)])  # 从4个动作中随机选择
        
        
    def learn(self):  # 学习方法
        if self.memory.trans_counter < self.batch_size: # wait before you start learning  # 等待缓冲区足够满再开始学习
            return
            
        # 1. Choose a sample from past transitions:  # 1. 从过去的转换中采样
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)  # 随机采样一批转换
        
        # 确保所有张量都在正确设备上
        states = ensure_tensor_on_device(states, device)  # 确保状态张量在正确设备上
        actions = ensure_tensor_on_device(actions, device)  # 确保动作张量在正确设备上
        rewards = ensure_tensor_on_device(rewards, device)  # 确保奖励张量在正确设备上
        new_states = ensure_tensor_on_device(new_states, device)  # 确保新状态张量在正确设备上
        terminals = ensure_tensor_on_device(terminals, device)  # 确保终止标志张量在正确设备上
        
        # 2. Update the target values  # 2. 更新目标值
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)  # 使用目标网络计算下一状态的Q值
        q_updated = rewards + self.gamma * q_next * (1 - terminals)  # 计算目标Q值（TD目标）
        q = self.q_func(states).gather(1, actions)  # 获取当前状态-动作对的Q值
        
        # 3. Update the main NN  # 3. 更新主网络
        loss = F.mse_loss(q, q_updated)  # 计算均方误差损失
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        
        # 4. Update the target NN (every N-th step)  # 4. 更新目标网络（每N步）
        if self.memory.trans_counter % self.replace_q_target == 0: # wait before you start learning  # 等待指定步数再更新目标网络
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):  # 遍历所有参数
                target_param.data.copy_(local_param.data)  # 将主网络参数复制到目标网络
                
        # 5. Reduce the exploration rate  # 5. 减少探索率
        self.reduce_epsilon()  # 调用减少探索率的方法
    
    def save_model(self, filepath, save_optimizer=True, save_memory=False):  # 保存模型的方法
        """保存训练好的模型
        
        Args:
            filepath (str): 模型保存路径
            save_optimizer (bool): 是否保存优化器状态
            save_memory (bool): 是否保存经验回放缓冲区
        """
        import os  # 导入操作系统接口模块
        import json  # 导入JSON数据处理模块
        
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # 创建目录，如果已存在则不报错
        
        # 保存模型状态
        model_state = {  # 创建模型状态字典
            'q_func_state_dict': self.q_func.state_dict(),  # 主网络状态字典
            'q_func_target_state_dict': self.q_func_target.state_dict(),  # 目标网络状态字典
            'observation_space_shape': self.q_func.fc1.in_features,  # 观察空间大小
            'action_space_n': self.q_func.fc3.out_features,  # 动作空间大小
            'gamma': self.gamma,  # 折扣因子
            'epsilon': self.epsilon,  # 当前探索率
            'epsilon_dec': self.epsilon_dec,  # 探索率衰减因子
            'epsilon_min': self.epsilon_min,  # 最小探索率
            'batch_size': self.batch_size,  # 批次大小
            'lr': self.optimizer.param_groups[0]['lr'],  # 学习率
            'replace_q_target': self.replace_q_target,  # 目标网络更新频率
            'mem_size': self.memory.memory_size,  # 内存缓冲区大小
            'device': str(device)  # 设备信息
        }
        
        # 保存优化器状态
        if save_optimizer:  # 如果需要保存优化器状态
            model_state['optimizer_state_dict'] = self.optimizer.state_dict()  # 保存优化器状态
        
        # 保存经验回放缓冲区
        if save_memory:  # 如果需要保存经验回放缓冲区
            model_state['memory_buffer'] = {  # 创建缓冲区信息字典
                'trans_counter': self.memory.trans_counter,  # 转换计数器
                'buffer_size': len(self.memory.buffer),  # 缓冲区大小
                'transitions': list(self.memory.buffer)  # 转换列表
            }
        
        # 保存模型
        torch.save(model_state, filepath)  # 使用PyTorch保存模型
        
        # 保存模型信息
        info_filepath = filepath.replace('.pth', '_info.json')  # 构造信息文件路径
        
        # 确保所有值都是JSON可序列化的
        def make_json_serializable(obj):  # 定义函数，将对象转换为JSON可序列化的格式
            """将对象转换为JSON可序列化的格式"""
            if isinstance(obj, np.integer):  # 如果是numpy整数
                return int(obj)  # 转换为Python整数
            elif isinstance(obj, np.floating):  # 如果是numpy浮点数
                return float(obj)  # 转换为Python浮点数
            elif isinstance(obj, np.ndarray):  # 如果是numpy数组
                return obj.tolist()  # 转换为Python列表
            elif isinstance(obj, torch.Tensor):  # 如果是PyTorch张量
                return obj.cpu().numpy().tolist()  # 转换为Python列表
            elif isinstance(obj, (list, tuple)):  # 如果是列表或元组
                return [make_json_serializable(item) for item in obj]  # 递归处理每个元素
            elif isinstance(obj, dict):  # 如果是字典
                return {key: make_json_serializable(value) for key, value in obj.items()}  # 递归处理每个值
            else:  # 其他类型
                return obj  # 直接返回
        
        # 准备模型信息
        model_info = {  # 创建模型信息字典
            'model_type': 'DDQN',  # 模型类型
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),  # 保存时间
            'parameters': {  # 参数信息
                'observation_space': make_json_serializable(model_state['observation_space_shape']),  # 观察空间
                'action_space': make_json_serializable(model_state['action_space_n']),  # 动作空间
                'gamma': make_json_serializable(model_state['gamma']),  # 折扣因子
                'epsilon': make_json_serializable(model_state['epsilon']),  # 探索率
                'learning_rate': make_json_serializable(model_state['lr'])  # 学习率
            }
        }
        
        with open(info_filepath, 'w', encoding='utf-8') as f:  # 以UTF-8编码打开文件
            json.dump(model_info, f, indent=2, ensure_ascii=False)  # 保存JSON信息
        
        print(f"✅ 模型已保存到: {filepath}")  # 打印保存成功信息
        print(f"📋 模型信息已保存到: {info_filepath}")  # 打印信息保存成功信息
    
    def load_model(self, filepath, load_optimizer=True, load_memory=False):  # 加载模型的方法
        """加载训练好的模型
        
        Args:
            filepath (str): 模型文件路径
            load_optimizer (bool): 是否加载优化器状态
            load_memory (bool): 是否加载经验回放缓冲区
        """
        if not os.path.exists(filepath):  # 如果文件不存在
            raise FileNotFoundError(f"模型文件不存在: {filepath}")  # 抛出文件不存在异常
        
        print(f"🔄 正在加载模型: {filepath}")  # 打印加载开始信息
        
        # 加载模型状态
        checkpoint = torch.load(filepath, map_location=device)  # 加载检查点文件到指定设备
        
        # 验证模型兼容性
        if checkpoint['observation_space_shape'] != self.q_func.fc1.in_features:  # 如果观察空间不匹配
            raise ValueError(f"观察空间不匹配: 期望 {self.q_func.fc1.in_features}, 得到 {checkpoint['observation_space_shape']}")  # 抛出值错误异常
        
        if checkpoint['action_space_n'] != self.q_func.fc3.out_features:  # 如果动作空间不匹配
            raise ValueError(f"动作空间不匹配: 期望 {self.q_func.fc3.out_features}, 得到 {checkpoint['action_space_n']}")  # 抛出值错误异常
        
        # 加载模型权重
        self.q_func.load_state_dict(checkpoint['q_func_state_dict'])  # 加载主网络权重
        self.q_func_target.load_state_dict(checkpoint['q_func_target_state_dict'])  # 加载目标网络权重
        
        # 加载优化器状态
        if load_optimizer and 'optimizer_state_dict' in checkpoint:  # 如果需要加载优化器状态且存在
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
            print("✅ 优化器状态已加载")  # 打印加载成功信息
        
        # 加载经验回放缓冲区
        if load_memory and 'memory_buffer' in checkpoint:  # 如果需要加载经验回放缓冲区且存在
            self.memory.trans_counter = checkpoint['memory_buffer']['trans_counter']  # 加载转换计数器
            self.memory.buffer = collections.deque(checkpoint['memory_buffer']['transitions'], 
                                                 maxlen=self.memory.memory_size)  # 加载转换缓冲区
            print("✅ 经验回放缓冲区已加载")  # 打印加载成功信息
        
        # 更新其他参数
        self.gamma = checkpoint['gamma']  # 更新折扣因子
        self.epsilon = checkpoint['epsilon']  # 更新探索率
        self.epsilon_dec = checkpoint['epsilon_dec']  # 更新探索率衰减因子
        self.epsilon_min = checkpoint['epsilon_min']  # 更新最小探索率
        self.batch_size = checkpoint['batch_size']  # 更新批次大小
        self.replace_q_target = checkpoint['replace_q_target']  # 更新目标网络更新频率
        
        print("✅ 模型加载完成!")  # 打印加载完成信息
        print(f"📊 模型信息:")  # 打印模型信息标题
        print(f"   - 观察空间: {checkpoint['observation_space_shape']}")  # 打印观察空间大小
        print(f"   - 动作空间: {checkpoint['action_space_n']}")  # 打印动作空间大小
        print(f"   - 当前探索率: {self.epsilon:.4f}")  # 打印当前探索率
        print(f"   - 折扣因子: {self.gamma}")  # 打印折扣因子
        print(f"   - 学习率: {checkpoint['lr']}")  # 打印学习率
    
    def export_onnx(self, filepath, input_shape=(1, 7)):  # 导出ONNX模型的方法
        """导出模型为ONNX格式，用于生产部署
        
        Args:
            filepath (str): ONNX文件保存路径
            input_shape (tuple): 输入张量形状
        """
        try:  # 尝试导入ONNX相关库
            import onnx  # 导入ONNX库
            import onnxruntime  # 导入ONNX运行时
        except ImportError:  # 如果导入失败
            print("❌ 需要安装onnx和onnxruntime: pip install onnx onnxruntime")  # 打印安装提示
            return False  # 返回失败标志
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape).to(device)  # 创建随机示例输入张量
        
        # 设置为评估模式
        self.q_func.eval()  # 设置为评估模式
        
        # 导出ONNX
        torch.onnx.export(  # 调用PyTorch的ONNX导出函数
            self.q_func,  # 要导出的模型
            dummy_input,  # 示例输入
            filepath,  # 输出文件路径
            export_params=True,  # 导出模型参数
            opset_version=11,  # ONNX操作集版本
            do_constant_folding=True,  # 执行常量折叠优化
            input_names=['input'],  # 输入名称
            output_names=['output'],  # 输出名称
            dynamic_axes={  # 动态轴设置
                'input': {0: 'batch_size'},  # 输入批次维度
                'output': {0: 'batch_size'}  # 输出批次维度
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(filepath)  # 加载ONNX模型
        onnx.checker.check_model(onnx_model)  # 检查模型有效性
        
        print(f"✅ ONNX模型已导出到: {filepath}")  # 打印导出成功信息
        print(f"📊 模型输入形状: {input_shape}")  # 打印输入形状
        print(f"🔍 ONNX模型验证通过")  # 打印验证通过信息
        
        return True  # 返回成功标志
    
    def get_model_info(self):  # 获取模型信息的方法
        """获取模型信息"""
        def make_json_serializable(obj):  # 定义函数，将对象转换为JSON可序列化的格式
            """将对象转换为JSON可序列化的格式"""
            if isinstance(obj, np.integer):  # 如果是numpy整数
                return int(obj)  # 转换为Python整数
            elif isinstance(obj, np.floating):  # 如果是numpy浮点数
                return float(obj)  # 转换为Python浮点数
            elif isinstance(obj, np.ndarray):  # 如果是numpy数组
                return obj.tolist()  # 转换为Python列表
            elif isinstance(obj, torch.Tensor):  # 如果是PyTorch张量
                return obj.cpu().numpy().tolist()  # 转换为Python列表
            elif isinstance(obj, (list, tuple)):  # 如果是列表或元组
                return [make_json_serializable(item) for item in obj]  # 递归处理每个元素
            elif isinstance(obj, dict):  # 如果是字典
                return {key: make_json_serializable(value) for key, value in obj.items()}  # 递归处理每个值
            else:  # 其他类型
                return obj  # 直接返回
        
        return {  # 返回模型信息字典
            'model_type': 'DDQN',  # 模型类型
            'observation_space': make_json_serializable(self.q_func.fc1.in_features),  # 观察空间大小
            'action_space': make_json_serializable(self.q_func.fc3.out_features),  # 动作空间大小
            'gamma': make_json_serializable(self.gamma),  # 折扣因子
            'epsilon': make_json_serializable(self.epsilon),  # 探索率
            'learning_rate': make_json_serializable(self.optimizer.param_groups[0]['lr']),  # 学习率
            'device': str(device),  # 设备信息
            'total_parameters': make_json_serializable(sum(p.numel() for p in self.q_func.parameters())),  # 总参数数量
            'trainable_parameters': make_json_serializable(sum(p.numel() for p in self.q_func.parameters() if p.requires_grad))  # 可训练参数数量
        }