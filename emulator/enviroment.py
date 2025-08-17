# 导入套接字模块
import socket
# 导入日志模块
import logging
# 导入时间模块
import time
# 导入队列模块
import queue

# 导入数值计算库
import numpy as np
# 导入强化学习环境库
import gymnasium as gym

# 导入空间定义
from gymnasium import spaces
# 导入默认字典
from collections import defaultdict

# 导入特征提取模块
from emulator import features
# 导入动作类型定义
from emulator.action_type import Action_type, Action

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class TrafficEnv(gym.Env):
    """交通环境类，继承自gym.Env，用于强化学习训练"""
    def __init__(self, proxy_host='127.0.0.1', proxy_port=8090,
                    server_host='127.0.0.1', server_port=8080,
                    mode='emulator', load_threshold=0.75, hazard_index=1):

        # 调用父类初始化
        super(TrafficEnv).__init__()
        # 设置运行模式
        self.mode = mode

        # 设置负载阈值
        self.load_threshold = load_threshold
        # 设置危险指数
        self.hazard_index = hazard_index
        # 请求缓冲区，存储(请求, 布尔值)，true表示来自正常用户的数据包
        self.request_buffer = []
        # 被阻止的IP地址集合
        self.blocked_ips = set()
        # 被阻止的IP地址块集合
        self.blocked_ip_blocks = set()

        # 定义观察空间（7维连续空间）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # 定义动作空间（4个离散动作）
        self.action_space = spaces.Discrete(4)

        # 初始化数据状态（5维）
        self.state_data = np.zeros(5, dtype=np.float32)
        # 初始化服务器状态（2维）
        self.state_server = np.zeros(2, dtype=np.float32)
        # 时间步计数器
        self.time_step = 0
        # 最大时间步数
        self.max_time_step = 25

        # 代理服务器配置
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        # 目标服务器配置
        self.server_host = server_host
        self.server_port = server_port

        # 创建特征提取器
        self.features = features.Features()

        # 创建代理服务器套接字
        self.proxy_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定代理服务器
        self.proxy_server.bind((self.proxy_host, self.proxy_port))
        # 开始监听
        self.proxy_server.listen()

        # 创建到目标服务器的连接
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.connect((self.server_host, self.server_port))

        # 当前数据
        self.current_data = None
        # 最后消息时间字典
        self.last_message_time = defaultdict(int)

        # 记录代理服务器启动信息
        logging.info(f"代理服务器已启动在 {self.proxy_host}:{self.proxy_port}")
        # 记录智能体连接信息
        logging.info(f"智能体已连接到服务器 {self.server_host}:{self.server_port}")


    def get_state(self):
        """获取当前状态的方法"""
        # 接受客户端连接
        conn, addr = self.proxy_server.accept()

        # 处理连接
        with conn:
            # 记录连接信息
            logging.info(f'来自 {addr} 的连接')
            # 接收数据
            data = conn.recv(1024).decode('utf-8')
            # 获取当前时间
            current_time = time.time()
            # 如果是模拟器模式，解析数据格式
            if self.mode=='emulator':
                data, addr, is_user = data.split('@@')
            
            # 如果有数据
            if data:
                # 保存当前数据
                self.current_data = data
                # 更新最后消息时间
                self.last_message_time[addr[0]] += current_time - self.last_message_time[addr[0]] 
                # 提取特征
                self.state_data = self.features.extract(data, self.last_message_time[addr[0]], addr[0])
            else:
                # 如果没有数据，使用零向量
                self.state_data = np.zeros(5, np.float32)
            
        # 获取服务器状态
        self.server.sendall(b'get state for features')
        data = self.server.recv(1024)
        # 解析CPU和内存使用率
        cpu_usage, memory_usage = data.decode('utf-8').split(' ')
        self.state_server = np.array([np.float32(cpu_usage), np.float32(memory_usage)])

        # 将请求添加到缓冲区
        self.request_buffer.append((addr,np.hstack((self.state_data, self.state_server)), bool(is_user)))
        
        return None


    def reset(self):
        """重置环境的方法"""
        # 重置数据状态
        self.state_data = np.zeros(5, dtype=np.float32)
        # 重置服务器状态
        self.state_server = np.zeros(2, dtype=np.float32)
        # 重置时间步
        self.time_step = 0
        # 清空请求缓冲区
        self.request_buffer = []

        # 获取25个初始状态
        for _ in range(25):
            self.get_state()
        
        # 打印缓冲区大小
        print(f'请求缓冲区大小: {len(self.request_buffer)}')

        # 返回第一个请求的状态
        return self.request_buffer[0][1]
    

    def step(self, action):
        """执行动作的方法"""
        # 时间步加1
        self.time_step += 1
        
        # 从缓冲区取出第一个请求
        addr,_,is_user = self.request_buffer.pop(0)
        
        # 根据动作类型执行相应操作
        if action == Action.SERVER_RECIEVE_CURRENT.value:
            # 接收当前请求，发送到服务器
            self.server.sendall(self.current_data.encode('utf-8'))
            logging.info(f'数据已发送到服务器 {self.server_host}:{self.server_port}')

        elif action == Action.SERVER_DROP_CURRENT.value:
            # 丢弃当前请求，不做任何操作
            pass

        elif action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value:
            # 阻止当前IP地址
            self.blocked_ips.add(addr)

        elif action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:
            # 阻止当前IP地址组
            request_source_block = self.features.get_netmask_from_ip(addr)
            self.blocked_ip_blocks.add(request_source_block)

        # 计算奖励
        reward = self.get_reward(action, addr, is_user)
        # 获取下一个状态
        state = self.request_buffer[0][1]
        # 判断是否结束
        done = self.time_step >= self.max_time_step
        # 额外信息
        info = {}
        

        return state, reward, done, info


    def get_server_usage(self):
        """获取服务器使用情况的方法"""
        # 发送获取服务器使用情况的请求
        self.server.sendall(b'get server usage')
        # 接收响应
        data = self.server.recv(1024)
        # 解析CPU和内存使用率
        cpu_usage, memory_usage, = data.decode('utf-8').split(' ')

        # 返回归一化的使用率
        return np.float32(cpu_usage)/100, np.float32(memory_usage)/100
    
    @staticmethod
    def get_action_type(action):
        """获取动作类型的静态方法"""
        # 如果是阻止IP地址组的动作，返回多目标动作类型
        if action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:
            return Action_type.MULTILPE_TARGET_ACTION.value
        else:
            # 否则返回单目标动作类型
            return Action_type.SINGLE_TARGET_ACTION.value

    def get_reward(self, action, addr, is_user):
        """计算奖励的方法"""
        # 获取服务器使用情况
        cpu_usage, memory_usage = self.get_server_usage()
        # 计算最大负载
        max_load = max(cpu_usage, memory_usage)
        # 判断是否重负载
        is_heavy_loaded = max_load >= self.load_threshold
        
        # 获取动作类型
        action_type = self.get_action_type(action)
        # 初始化奖励
        reward = 0

        # 如果服务器重负载
        if is_heavy_loaded:
            if action_type == Action_type.SINGLE_TARGET_ACTION.value:
                # 如果是用户且接收请求（真阴性）
                if is_user == True and action == Action.SERVER_RECIEVE_CURRENT.value:
                    return reward
                
                # 如果是用户但丢弃或阻止（假阳性）
                elif is_user == True and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value):
                    reward = -2/(max_load / self.load_threshold * self.hazard_index)
                    return reward
                
                # 如果不是用户但接收请求（假阴性）
                elif is_user == False and action == Action.SERVER_RECIEVE_CURRENT.value:
                    reward = -(max_load / self.load_threshold * self.hazard_index)
                    return reward
                
                # 如果不是用户且丢弃或阻止（真阳性）
                elif is_user == False and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value):
                    reward = max_load / self.load_threshold * self.hazard_index
                    return reward
                
                else:
                    return ValueError(f"未知动作: {action}")
                
            elif action_type == Action_type.MULTILPE_TARGET_ACTION.value:
                # 如果是阻止IP地址组的动作
                if action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:
                    # 初始化假阳性和真阳性计数
                    fp, tp = 0, 0
                    # 获取目标地址组
                    target_adress_group = self.features.get_netmask_from_ip(addr)
                    # 遍历缓冲区中的其他请求
                    for i in range(1, len(self.request_buffer)):
                        # 如果请求来自同一地址组
                        if self.features.get_netmask_from_ip(self.request_buffer[i][0]) == target_adress_group:
                            # 如果是用户请求，增加假阳性计数
                            if self.request_buffer[i][2] == True:
                                fp+=1
                            # 如果不是用户请求，增加真阳性计数
                            else:
                                tp+=1
                    
                    # 计算奖励：真阳性奖励 - 假阳性惩罚
                    reward = (max_load / self.load_threshold * self.hazard_index) * tp - (2 / (max_load / self.load_threshold * self.hazard_index) * fp)
                    return reward

                else:
                    return ValueError(f"未知动作: {action}")
        
        else:
            # 如果服务器不是重负载
            if action_type == Action_type.SINGLE_TARGET_ACTION.value:
                # 如果是用户且接收请求（真阴性）
                if is_user == True and action == Action.SERVER_RECIEVE_CURRENT.value:
                    return reward
                
                # 如果是用户但丢弃或阻止（假阳性）
                elif is_user == True and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value):
                    reward = -2
                    return reward
                
                # 如果不是用户但接收请求（假阴性）
                elif is_user == False and action == Action.SERVER_RECIEVE_CURRENT.value:
                    reward = -1
                    return reward
                
                # 如果不是用户且丢弃或阻止（真阳性）
                elif is_user == False and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value):
                    reward = 1
                    return reward
                
                else:
                    return ValueError(f"未知动作: {action}")
                
            elif action_type == Action_type.MULTILPE_TARGET_ACTION.value:
                # 如果是阻止IP地址组的动作
                if action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:
                    # 初始化假阳性和真阳性计数
                    fp, tp = 0, 0
                    # 获取目标地址组
                    target_adress_group = self.features.get_netmask_from_ip(addr)
                    # 遍历缓冲区中的其他请求
                    for i in range(1, len(self.request_buffer)):
                        # 如果请求来自同一地址组
                        if self.features.get_netmask_from_ip(self.request_buffer[i][0]) == target_adress_group:
                            # 如果是用户请求，增加假阳性计数
                            if self.request_buffer[i][2] == True:
                                fp+=1
                            # 如果不是用户请求，增加真阳性计数
                            else:
                                tp+=1
                    
                    # 计算奖励：真阳性奖励 - 假阳性惩罚
                    reward = tp - 2 * fp
                    return reward

                else:
                    return ValueError(f"未知动作: {action}")
                