import socket  # 导入套接字模块，用于网络通信
import logging  # 导入日志模块，用于记录程序运行信息
import time  # 导入时间模块，用于时间相关操作
import queue  # 导入队列模块，用于数据缓冲

import numpy as np  # 导入数值计算库，用于数组操作和数学计算
import gymnasium as gym  # 导入强化学习环境库，用于创建标准化的强化学习环境

from gymnasium import spaces  # 导入空间模块，用于定义观察空间和动作空间
from collections import defaultdict  # 导入默认字典，用于自动初始化字典值

from emulator import features  # 导入特征提取模块
from emulator.action_type import Action_type, Action  # 导入动作类型和动作枚举

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')  # 配置日志格式和级别

class TrafficEnv(gym.Env):  # 定义流量环境类，继承自Gymnasium环境基类
    def __init__(self, proxy_host='127.0.0.1', proxy_port=8090,  # 初始化方法，设置代理服务器和服务器参数
                    server_host='127.0.0.1', server_port=8080,
                    mode='emulator', load_threshold=0.75, hazard_index=1):

        super(TrafficEnv).__init__()  # 调用父类初始化方法
        self.mode = mode  # 设置运行模式（模拟器或真实环境）

        self.load_threshold = load_threshold  # 设置负载阈值，用于判断服务器是否过载
        self.hazard_index = hazard_index  # 设置危险指数，用于调整奖励计算
        self.request_buffer = [] # array element (request, bool), true if packet is from normal user  # 请求缓冲区，存储请求和用户类型信息
        self.blocked_ips = set()  # 被阻止的IP地址集合
        self.blocked_ip_blocks = set()  # 被阻止的IP地址块集合

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # 定义观察空间为7维连续空间
        self.action_space = spaces.Discrete(4)  # 定义动作空间为4个离散动作

        self.state_data = np.zeros(5, dtype=np.float32)  # 初始化数据状态数组（5维）
        self.state_server = np.zeros(2, dtype=np.float32)  # 初始化服务器状态数组（2维）
        self.time_step = 0  # 初始化时间步计数器
        self.max_time_step = 25  # 设置最大时间步数

        self.proxy_host = proxy_host  # 代理服务器主机地址
        self.proxy_port = proxy_port  # 代理服务器端口
        self.server_host = server_host  # 目标服务器主机地址
        self.server_port = server_port  # 目标服务器端口

        self.features = features.Features()  # 创建特征提取器实例

        self.proxy_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建TCP套接字
        self.proxy_server.bind((self.proxy_host, self.proxy_port))  # 绑定代理服务器地址和端口
        self.proxy_server.listen()  # 开始监听连接

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建连接到目标服务器的套接字
        self.server.connect((self.server_host, self.server_port))  # 连接到目标服务器

        self.current_data = None  # 当前数据包内容
        self.last_message_time = defaultdict(int)  # 记录每个IP地址最后消息时间的字典

        logging.info(f"Proxy server started on {self.proxy_host}:{self.proxy_port}")  # 记录代理服务器启动信息
        logging.info(f"Agent connected to server at {self.server_host}:{self.server_port}")  # 记录连接到目标服务器的信息


    def get_state(self):  # 获取当前状态的方法
        conn, addr = self.proxy_server.accept()  # 接受客户端连接，返回连接对象和地址

        with conn:  # 使用连接上下文管理器
            logging.info(f'Connection from {addr}')  # 记录连接来源信息
            data = conn.recv(1024).decode('utf-8')  # 接收数据并解码
            current_time = time.time()  # 获取当前时间
            if self.mode=='emulator':  # 如果是模拟器模式
                data, addr, is_user = data.split('@@')  # 解析数据格式：数据@@地址@@用户类型
            
            if data:  # 如果接收到数据
                self.current_data = data  # 保存当前数据
                self.last_message_time[addr[0]] += current_time - self.last_message_time[addr[0]]  # 更新最后消息时间
                self.state_data = self.features.extract(data, self.last_message_time[addr[0]], addr[0])  # 提取特征
            else:  # 如果没有接收到数据
                self.state_data = np.zeros(5, np.float32)  # 将状态数据设为零向量
            
        self.server.sendall(b'get state for features')  # 向目标服务器发送获取状态请求
        data = self.server.recv(1024)  # 接收服务器响应
        cpu_usage, memory_usage = data.decode('utf-8').split(' ')  # 解析CPU和内存使用率
        self.state_server = np.array([np.float32(cpu_usage), np.float32(memory_usage)])  # 转换为numpy数组

        self.request_buffer.append((addr,np.hstack((self.state_data, self.state_server)), bool(is_user)))  # 将请求信息添加到缓冲区
        
        return None  # 返回None（此方法主要用于更新状态）


    def reset(self):  # 重置环境的方法
        self.state_data = np.zeros(5, dtype=np.float32)  # 重置数据状态为零向量
        self.state_server = np.zeros(2, dtype=np.float32)  # 重置服务器状态为零向量
        self.time_step = 0  # 重置时间步计数器
        self.request_buffer = []  # 清空请求缓冲区

        for _ in range(25):  # 循环25次
            self.get_state()  # 获取状态，填充缓冲区
        
        print(f'Shape of request_buffer: {len(self.request_buffer)}')  # 打印缓冲区大小

        return self.request_buffer[0][1]  # 返回第一个请求的状态
    

    def step(self, action):  # 执行动作的方法
        self.time_step += 1  # 增加时间步计数器
        
        addr,_,is_user = self.request_buffer.pop(0)  # 从缓冲区取出第一个请求的信息
        
        if action == Action.SERVER_RECIEVE_CURRENT.value:  # 如果动作是接收当前请求
            self.server.sendall(self.current_data.encode('utf-8'))  # 将数据发送到目标服务器
            logging.info(f'Data sent to server {self.server_host}:{self.server_port}')  # 记录发送信息

        elif action == Action.SERVER_DROP_CURRENT.value:  # 如果动作是丢弃当前请求
            pass  # 不执行任何操作

        elif action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value:  # 如果动作是阻止当前IP地址
            self.blocked_ips.add(addr)  # 将IP地址添加到阻止列表

        elif action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:  # 如果动作是阻止当前IP地址组
            request_source_block = self.features.get_netmask_from_ip(addr)  # 获取IP地址的网络掩码
            self.blocked_ip_blocks.add(request_source_block)  # 将网络掩码添加到阻止列表

        reward = self.get_reward(action, addr, is_user)  # 计算奖励
        state = self.request_buffer[0][1]  # 获取下一个状态
        done = self.time_step >= self.max_time_step  # 判断是否结束
        info = {}  # 额外信息字典
        

        return state, reward, done, info  # 返回状态、奖励、结束标志和信息


    def get_server_usage(self):  # 获取服务器使用率的方法
        self.server.sendall(b'get server usage')  # 向服务器发送获取使用率请求
        data = self.server.recv(1024)  # 接收响应
        cpu_usage, memory_usage, = data.decode('utf-8').split(' ')  # 解析CPU和内存使用率

        return np.float32(cpu_usage)/100, np.float32(memory_usage)/100  # 返回归一化的使用率（0-1之间）
    
    @staticmethod  # 静态方法装饰器
    def get_action_type(action):  # 获取动作类型的方法
        if action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:  # 如果是阻止IP地址组的动作
            return Action_type.MULTILPE_TARGET_ACTION.value  # 返回多目标动作类型
        else:  # 否则
            return Action_type.SINGLE_TARGET_ACTION.value  # 返回单目标动作类型

    def get_reward(self, action, addr, is_user):  # 计算奖励的方法
        cpu_usage, memory_usage = self.get_server_usage()  # 获取服务器使用率
        max_load = max(cpu_usage, memory_usage)  # 取CPU和内存使用率的较大值
        is_heavy_loaded = max_load >= self.load_threshold  # 判断是否过载
        
        action_type = self.get_action_type(action)  # 获取动作类型
        reward = 0  # 初始化奖励为0

        if is_heavy_loaded:  # 如果服务器过载
            if action_type == Action_type.SINGLE_TARGET_ACTION.value:  # 如果是单目标动作
                if is_user == True and action == Action.SERVER_RECIEVE_CURRENT.value: #True negative  # 真阴性：正常用户被接受
                    return reward  # 返回0奖励
                
                elif is_user == True and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value): #False Positive  # 假阳性：正常用户被阻止
                    reward = -2/(max_load / self.load_threshold * self.hazard_index)  # 计算负奖励
                    return reward  # 返回负奖励
                
                elif is_user == False and action == Action.SERVER_RECIEVE_CURRENT.value: #False Negative  # 假阴性：恶意用户被接受
                    reward = -(max_load / self.load_threshold * self.hazard_index)  # 计算负奖励
                    return reward  # 返回负奖励
                
                elif is_user == False and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value): #True Positive  # 真阳性：恶意用户被阻止
                    reward = max_load / self.load_threshold * self.hazard_index  # 计算正奖励
                    return reward  # 返回正奖励
                
                else:  # 其他情况
                    return ValueError(f"Unknown action: {action}")  # 抛出值错误异常
                
            elif action_type == Action_type.MULTILPE_TARGET_ACTION.value:  # 如果是多目标动作
                if action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:  # 如果是阻止IP地址组
                    fp, tp = 0, 0  # 初始化假阳性和真阳性计数器
                    target_adress_group = self.features.get_netmask_from_ip(addr)  # 获取目标IP地址组
                    for i in range(1, len(self.request_buffer)):  # 遍历缓冲区中的其他请求
                        if self.features.get_netmask_from_ip(self.request_buffer[i][0]) == target_adress_group:  # 如果IP地址属于同一组
                            if self.request_buffer[i][2] == True:  # 如果是正常用户
                                fp+=1  # 假阳性计数加1

                            else:  # 如果是恶意用户
                                tp+=1  # 真阳性计数加1
                    
                    reward = (max_load / self.load_threshold * self.hazard_index) * tp - (2 / (max_load / self.load_threshold * self.hazard_index) * fp)  # 计算综合奖励
                    return reward  # 返回奖励

                else:  # 其他多目标动作
                    return ValueError(f"Unknown action: {action}")  # 抛出值错误异常
        
        else:  # 如果服务器未过载
            if action_type == Action_type.SINGLE_TARGET_ACTION.value:  # 如果是单目标动作
                if is_user == True and action == Action.SERVER_RECIEVE_CURRENT.value: #True negative  # 真阴性：正常用户被接受
                    return reward  # 返回0奖励
                
                elif is_user == True and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value): #False Positive  # 假阳性：正常用户被阻止
                    reward = -2  # 固定负奖励
                    return reward  # 返回负奖励
                
                elif is_user == False and action == Action.SERVER_RECIEVE_CURRENT.value: #False Negative  # 假阴性：恶意用户被接受
                    reward = -1  # 固定负奖励
                    return reward  # 返回负奖励
                
                elif is_user == False and (action == Action.SERVER_DROP_CURRENT.value or action == Action.SERVER_BLOCK_CURRENT_ADDRESS.value): #True Positive  # 真阳性：恶意用户被阻止
                    reward = 1  # 固定正奖励
                    return reward  # 返回正奖励
                
                else:  # 其他情况
                    return ValueError(f"Unknown action: {action}")  # 抛出值错误异常
                
            elif action_type == Action_type.MULTILPE_TARGET_ACTION.value:  # 如果是多目标动作
                if action == Action.SERVER_BLOCK_CURRENT_ADDRESS_GROUP.value:  # 如果是阻止IP地址组
                    fp, tp = 0, 0  # 初始化假阳性和真阳性计数器
                    target_adress_group = self.features.get_netmask_from_ip(addr)  # 获取目标IP地址组
                    for i in range(1, len(self.request_buffer)):  # 遍历缓冲区中的其他请求
                        if self.features.get_netmask_from_ip(self.request_buffer[i][0]) == target_adress_group:  # 如果IP地址属于同一组
                            if self.request_buffer[i][2] == True:  # 如果是正常用户
                                fp+=1  # 假阳性计数加1

                            else:  # 如果是恶意用户
                                tp+=1  # 真阳性计数加1
                    
                    reward = tp - 2 * fp  # 计算综合奖励（真阳性 - 2*假阳性）
                    return reward  # 返回奖励

                else:  # 其他多目标动作
                    return ValueError(f"Unknown action: {action}")  # 抛出值错误异常
                