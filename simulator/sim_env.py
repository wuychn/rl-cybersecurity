# 导入强化学习环境库
import gymnasium as gym
# 导入空间定义
from gymnasium import spaces
# 导入数值计算库
import numpy as np
# 导入IP地址处理模块
import ipaddress
# 导入双端队列和默认字典
from collections import deque, defaultdict

class HTTPServerEnv(gym.Env):
    """HTTP服务器环境类，继承自gym.Env，用于模拟HTTP服务器流量"""
    def __init__(self, buffer_size=100, user_ips=None, load_threshold=0.8, hazard_index=1, user_message_sizes=None, user_request_prob=0.2):
        # 调用父类初始化
        super(HTTPServerEnv, self).__init__()
        
        # 缓冲区和用户地址参数
        self.buffer_size = buffer_size
        self.user_ips = set(user_ips) if user_ips else set(self.generate_random_ips(250))
        
        # 负载和危险指数参数
        self.load_threshold = load_threshold
        self.hazard_index = hazard_index
        
        # 用户地址消息大小的固定池
        self.user_message_sizes = user_message_sizes if user_message_sizes else [100, 200, 300, 400, 500]
        
        # 添加用户请求的概率
        self.user_request_prob = user_request_prob
        
        # 定义状态空间（7个元素）
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000000, 1000000, 1000, 1000, 10000, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 0 - 接受, 1 - 拒绝, 2 - 阻止源, 3 - 阻止组
        
        # 模拟参数
        self.max_messages = 700
        self.current_message_count = 0
        self.blocked_ips = set()
        self.blocked_groups = set()
        
        # 源地址统计
        self.source_stats = defaultdict(lambda: {
            "last_time": 0,
            "intervals": [],
            "total_messages": 0,
        })
        
        # 按组划分的流量大小
        self.group_traffic = defaultdict(int)
        
        # 请求缓冲区（存储state, source_ip, is_user_request）
        self.request_buffer = deque(maxlen=self.buffer_size)

    def reset(self):
        """重置环境到初始状态"""
        # 重置消息计数
        self.current_message_count = 0
        # 清空被阻止的IP地址
        self.blocked_ips = set()
        # 清空被阻止的组
        self.blocked_groups = set()
        # 重置源地址统计
        self.source_stats = defaultdict(lambda: {
            "last_time": 0,
            "intervals": [],
            "total_messages": 0,
        })
        # 重置组流量
        self.group_traffic = defaultdict(int)
        # 清空请求缓冲区
        self.request_buffer.clear()
        # 初始填充缓冲区
        self.fill_buffer()
        # 返回第一个请求的状态
        return self.request_buffer[0][0]

    def step(self, action):
        """执行动作的方法"""
        # 模拟请求处理
        done = False
        info = {}
        
        # 从缓冲区获取当前请求并移除
        state, source_ip, is_user_request = self.request_buffer.popleft()
        
        # 更新源地址统计
        current_time = self.current_message_count
        source_stat = self.source_stats[source_ip]
        if source_stat["last_time"] > 0:
            # 计算时间间隔
            interval = current_time - source_stat["last_time"]
            source_stat["intervals"].append(interval)
        source_stat["last_time"] = current_time
        source_stat["total_messages"] += 1
        
        # 计算平均时间和偏差
        if len(source_stat["intervals"]) > 0:
            avg_time = np.mean(source_stat["intervals"])
            avg_abs_deviation = np.mean(np.abs(np.array(source_stat["intervals"]) - avg_time))
        else:
            avg_time = 0
            avg_abs_deviation = 0
        
        # 更新组流量大小
        group = self.get_source_group(source_ip)
        if is_user_request:
            # 用户请求使用固定大小
            message_size = np.random.choice(self.user_message_sizes)
        else:
            # 非用户请求使用随机大小
            message_size = np.random.randint(100, 1000000)
        self.group_traffic[group] += message_size
        
        # 随机CPU和内存负载
        cpu_load = np.random.uniform(0, 1)
        memory_load = np.random.uniform(0, 1)
        
        # 使用generate_random_state创建新状态
        new_state, _, _ = self.generate_random_state()
        
        # 计算奖励
        reward = self.get_reward(action, is_user_request, source_ip, group, cpu_load, memory_load)
        
        # 处理动作：阻止IP地址或组
        if action == 2:  # 阻止源地址
            self.blocked_ips.add(source_ip)
        elif action == 3:  # 阻止组
            self.blocked_groups.add(group)
        
        # 从缓冲区移除被阻止的请求
        self.remove_blocked_requests()
        
        # 向缓冲区添加新请求
        self.add_request_to_buffer(new_state)
        
        # 如果缓冲区在移除被阻止请求后变空，则补充到100个元素
        while len(self.request_buffer) < self.buffer_size:
            self.add_request_to_buffer(self.generate_random_state()[0])
        
        # 检查结束条件
        self.current_message_count += 1
        if self.current_message_count >= self.max_messages:
            done = True
        
        # 返回新状态（缓冲区的第零个元素，不移除）
        next_state = self.request_buffer[0][0]
        return next_state, reward, done, info

    def get_reward(self, action, is_user_request, source_ip, group, cpu_load, memory_load):
        """根据动作计算奖励"""
        # 计算最大负载
        max_load = max(cpu_load, memory_load)
        
        # 如果负载超过阈值
        if max_load > self.load_threshold:
            if action in [2, 3]:  # 阻止源地址或组
                tp, fp = self.count_requests_in_buffer(group)
                return (max_load / self.load_threshold * self.hazard_index) * tp - (2 / (max_load / self.load_threshold * self.hazard_index) * fp)
            else:  # 动作影响1个请求
                if is_user_request and action == 0:  # 接受用户
                    return 0
                elif is_user_request and action in [1, 2]:  # 拒绝或阻止用户
                    return -2 / (max_load / self.load_threshold * self.hazard_index)
                elif not is_user_request and action == 0:  # 接受攻击者
                    return -(max_load / self.load_threshold * self.hazard_index)
                elif not is_user_request and action in [1, 2]:  # 拒绝或阻止攻击者
                    return max_load / self.load_threshold * self.hazard_index
        else:
            # 如果负载未超过阈值
            if action in [2, 3]:  # 阻止源地址或组
                tp, fp = self.count_requests_in_buffer(group)
                return tp - 2 * fp
            else:  # 动作影响1个请求
                if is_user_request and action == 0:  # 接受用户
                    return 0
                elif is_user_request and action in [1, 2]:  # 拒绝或阻止用户
                    return -2
                elif not is_user_request and action == 0:  # 接受攻击者
                    return -1
                elif not is_user_request and action in [1, 2]:  # 拒绝或阻止攻击者
                    return 1

    def count_requests_in_buffer(self, group):
        """计算缓冲区中用户和攻击者请求的数量"""
        tp = 0  # 真阳性（来自用户的请求）
        fp = 0  # 假阳性（来自攻击者的请求）
        for state, ip, is_user_request in self.request_buffer:
            if self.get_source_group(ip) == group:
                if is_user_request:
                    tp += 1
                else:
                    fp += 1
        return tp, fp

    def generate_random_ip(self):
        """生成随机IPv4地址，排除特殊地址"""
        while True:
            ip = str(ipaddress.IPv4Address(np.random.randint(0, 256**4)))
            if ip not in ["127.0.0.1", "0.0.0.0"]:
                return ip

    def generate_random_ips(self, count):
        """生成指定数量的随机IPv4地址列表"""
        ips = set()
        while len(ips) < count:
            ips.add(self.generate_random_ip())
        return list(ips)

    def get_source_group(self, source_ip):
        """按/24子网进行分组"""
        ip = ipaddress.IPv4Address(source_ip)
        return str(ipaddress.IPv4Network(f"{ip}/24", strict=False).network_address)

    def generate_random_state(self):
        """生成随机状态"""
        # 确定请求是否为用户请求
        is_user_request = np.random.rand() < self.user_request_prob
        
        # 根据请求类型生成IP地址
        if is_user_request:
            source_ip = np.random.choice(list(self.user_ips))
        else:
            while True:
                source_ip = self.generate_random_ip()
                if source_ip not in self.user_ips:
                    break
        
        # 生成消息大小
        if is_user_request:
            message_size = np.random.choice(self.user_message_sizes)
        else:
            message_size = np.random.randint(100, 1000000)
        
        # 按/24子网分组
        group = self.get_source_group(source_ip)
        
        # 更新组的流量大小
        self.group_traffic[group] += message_size
        
        # 随机CPU和内存负载
        cpu_load = np.random.uniform(0, 1)
        memory_load = np.random.uniform(0, 1)
        
        # 生成源地址统计
        source_stat = self.source_stats[source_ip]
        if source_stat["last_time"] > 0:
            avg_time = np.mean(source_stat["intervals"]) if source_stat["intervals"] else 0
            avg_abs_deviation = np.mean(np.abs(np.array(source_stat["intervals"]) - avg_time)) if source_stat["intervals"] else 0
        else:
            avg_time = 0
            avg_abs_deviation = 0
        
        # 创建完整状态
        state = np.array([
            message_size,
            self.group_traffic[group],
            avg_time,
            avg_abs_deviation,
            source_stat["total_messages"],
            cpu_load,
            memory_load,
        ], dtype=np.float32)
        
        return state, source_ip, is_user_request

    def fill_buffer(self):
        """填充缓冲区到指定大小"""
        while len(self.request_buffer) < self.buffer_size:
            state, source_ip, is_user_request = self.generate_random_state()
            self.request_buffer.append((state, source_ip, is_user_request))

    def add_request_to_buffer(self, state):
        """向缓冲区添加请求"""
        # 确定请求类别：用户或攻击者
        is_user_request = np.random.rand() < self.user_request_prob
        
        # 根据请求类别生成IP地址
        while True:
            if is_user_request:
                source_ip = np.random.choice(list(self.user_ips))
            else:
                source_ip = self.generate_random_ip()
                if source_ip not in self.user_ips:
                    break
            
            # 检查IP地址及其组是否未被阻止
            if source_ip not in self.blocked_ips and self.get_source_group(source_ip) not in self.blocked_groups:
                break
        
        # 将请求添加到缓冲区
        self.request_buffer.append((state, source_ip, is_user_request))

    def remove_blocked_requests(self):
        """移除属于被阻止IP地址或组的请求"""
        self.request_buffer = deque(
            (state, ip, is_user) for state, ip, is_user in self.request_buffer
            if ip not in self.blocked_ips and self.get_source_group(ip) not in self.blocked_groups
        )

    def render(self, mode='human'):
        pass