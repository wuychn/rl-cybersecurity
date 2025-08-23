import gymnasium as gym  # 导入强化学习环境库，用于创建标准化的强化学习环境
from gymnasium import spaces  # 导入空间模块，用于定义观察空间和动作空间
import numpy as np  # 导入数值计算库，用于数组操作和数学计算
import ipaddress  # 导入IP地址处理模块，用于IP地址操作和网络分组
from collections import deque, defaultdict  # 导入双端队列和默认字典，用于数据缓冲和统计

class HTTPServerEnv(gym.Env):  # 定义HTTP服务器环境类，继承自Gymnasium环境基类
    def __init__(self, buffer_size=100, user_ips=None, load_threshold=0.8, hazard_index=1, user_message_sizes=None, user_request_prob=0.2):  # 初始化方法，设置各种参数
        super(HTTPServerEnv, self).__init__()  # 调用父类初始化方法
        
        # Buffer and user address parameters  # 缓冲区和用户地址参数
        self.buffer_size = buffer_size  # 设置缓冲区大小
        self.user_ips = set(user_ips) if user_ips else set(self.generate_random_ips(250))  # 设置用户IP地址集合，如果没有提供则生成250个随机IP
        
        # Load and hazard index parameters  # 负载和危险指数参数
        self.load_threshold = load_threshold  # 设置负载阈值
        self.hazard_index = hazard_index  # 设置危险指数
        
        # Pool of fixed message sizes for user addresses  # 用户地址的固定消息大小池
        self.user_message_sizes = user_message_sizes if user_message_sizes else [100, 200, 300, 400, 500]  # 设置用户消息大小池
        
        # Probability of adding a user request  # 添加用户请求的概率
        self.user_request_prob = user_request_prob  # 设置用户请求概率
        
        # Define state space (7 elements)  # 定义状态空间（7个元素）
        self.observation_space = spaces.Box(  # 创建连续状态空间
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # 状态空间下界
            high=np.array([1000000, 1000000, 1000, 1000, 10000, 1, 1], dtype=np.float32),  # 状态空间上界
            dtype=np.float32  # 数据类型
        )
        
        # Define action space  # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 0 - accept, 1 - reject, 2 - block source, 3 - block group  # 创建离散动作空间，4个动作：接受、拒绝、阻止源、阻止组
        
        # Simulation parameters  # 模拟参数
        self.max_messages = 700  # 设置最大消息数量
        self.current_message_count = 0  # 初始化当前消息计数器
        self.blocked_ips = set()  # 创建被阻止的IP地址集合
        self.blocked_groups = set()  # 创建被阻止的IP地址组集合
        
        # Source statistics  # 源地址统计信息
        self.source_stats = defaultdict(lambda: {  # 创建默认字典，存储每个源地址的统计信息
            "last_time": 0,  # 最后访问时间
            "intervals": [],  # 访问间隔列表
            "total_messages": 0,  # 总消息数量
        })
        
        # Traffic size by groups  # 按组分类的流量大小
        self.group_traffic = defaultdict(int)  # 创建默认字典，存储每个组的流量大小
        
        # Request buffer (stores state, source_ip, is_user_request)  # 请求缓冲区（存储状态、源IP、是否为用户请求）
        self.request_buffer = deque(maxlen=self.buffer_size)  # 创建双端队列作为请求缓冲区

    def reset(self):  # 重置环境到初始状态的方法
        # Reset environment to initial state  # 重置环境到初始状态
        self.current_message_count = 0  # 重置当前消息计数器
        self.blocked_ips = set()  # 清空被阻止的IP地址集合
        self.blocked_groups = set()  # 清空被阻止的IP地址组集合
        self.source_stats = defaultdict(lambda: {  # 重置源地址统计信息
            "last_time": 0,  # 重置最后访问时间
            "intervals": [],  # 清空访问间隔列表
            "total_messages": 0,  # 重置总消息数量
        })
        self.group_traffic = defaultdict(int)  # 重置组流量统计
        self.request_buffer.clear()  # 清空请求缓冲区
        self.fill_buffer()  # Primary buffer filling  # 主要缓冲区填充
        return self.request_buffer[0][0]  # 返回缓冲区第一个请求的状态

    def step(self, action):  # 执行动作的方法
        # Simulate request processing  # 模拟请求处理
        done = False  # 初始化完成标志
        info = {}  # 初始化信息字典
        
        # Get current request from buffer with removal  # 从缓冲区获取当前请求并移除
        state, source_ip, is_user_request = self.request_buffer.popleft()  # 从缓冲区左侧取出请求信息
        
        # Update source statistics  # 更新源地址统计信息
        current_time = self.current_message_count  # 获取当前时间
        source_stat = self.source_stats[source_ip]  # 获取源地址的统计信息
        if source_stat["last_time"] > 0:  # 如果之前有访问记录
            interval = current_time - source_stat["last_time"]  # 计算访问间隔
            source_stat["intervals"].append(interval)  # 将间隔添加到列表中
        source_stat["last_time"] = current_time  # 更新最后访问时间
        source_stat["total_messages"] += 1  # 增加总消息数量
        
        # Calculate average time and deviation  # 计算平均时间和偏差
        if len(source_stat["intervals"]) > 0:  # 如果有访问间隔数据
            avg_time = np.mean(source_stat["intervals"])  # 计算平均访问间隔
            avg_abs_deviation = np.mean(np.abs(np.array(source_stat["intervals"]) - avg_time))  # 计算平均绝对偏差
        else:  # 如果没有访问间隔数据
            avg_time = 0  # 平均时间设为0
            avg_abs_deviation = 0  # 平均绝对偏差设为0
        
        # Update traffic size by group  # 按组更新流量大小
        group = self.get_source_group(source_ip)  # 获取源IP地址的组
        if is_user_request:  # 如果是用户请求
            message_size = np.random.choice(self.user_message_sizes)  # 从用户消息大小池中随机选择
        else:  # 如果是攻击者请求
            message_size = np.random.randint(100, 1000000)  # 随机生成100到1000000之间的消息大小
        self.group_traffic[group] += message_size  # 累加该组的流量大小
        
        # Random CPU and memory load  # 随机CPU和内存负载
        cpu_load = np.random.uniform(0, 1)  # 生成0到1之间的随机CPU负载
        memory_load = np.random.uniform(0, 1)  # 生成0到1之间的随机内存负载
        
        # Create new state using generate_random_state  # 使用generate_random_state创建新状态
        new_state, _, _ = self.generate_random_state()  # 生成新的随机状态
        
        # Calculate reward  # 计算奖励
        reward = self.get_reward(action, is_user_request, source_ip, group, cpu_load, memory_load)  # 根据动作和状态计算奖励
        
        # Handle actions: block IP address or group  # 处理动作：阻止IP地址或组
        if action == 2:  # Block source  # 阻止源地址
            self.blocked_ips.add(source_ip)  # 将源IP地址添加到阻止列表
        elif action == 3:  # Block group  # 阻止组
            self.blocked_groups.add(group)  # 将IP地址组添加到阻止列表
        
        # Remove blocked requests from buffer  # 从缓冲区移除被阻止的请求
        self.remove_blocked_requests()  # 调用移除被阻止请求的方法
        
        # Add new request to buffer  # 向缓冲区添加新请求
        self.add_request_to_buffer(new_state)  # 调用添加请求到缓冲区的方法
        
        # Fill buffer to 100 elements if it emptied after removing blocked requests  # 如果移除被阻止请求后缓冲区空了，则填充到100个元素
        while len(self.request_buffer) < self.buffer_size:  # 当缓冲区大小小于设定值时
            self.add_request_to_buffer(self.generate_random_state()[0])  # 添加随机生成的状态
        
        # Check completion condition  # 检查完成条件
        self.current_message_count += 1  # 增加当前消息计数
        if self.current_message_count >= self.max_messages:  # 如果达到最大消息数量
            done = True  # 设置完成标志为True
        
        # Return new state (zeroth buffer element without removal)  # 返回新状态（缓冲区第0个元素，不移除）
        next_state = self.request_buffer[0][0]  # 获取下一个状态
        return next_state, reward, done, info  # 返回下一个状态、奖励、完成标志和信息

    def get_reward(self, action, is_user_request, source_ip, group, cpu_load, memory_load):  # 计算奖励的方法
        # Calculate reward based on action  # 根据动作计算奖励
        max_load = max(cpu_load, memory_load)  # 取CPU和内存负载的较大值
        
        if max_load > self.load_threshold:  # 如果负载超过阈值
            if action in [2, 3]:  # Block source or group  # 阻止源地址或组
                tp, fp = self.count_requests_in_buffer(group)  # 统计缓冲区中该组的真阳性和假阳性
                return (max_load / self.load_threshold * self.hazard_index) * tp - (2 / (max_load / self.load_threshold * self.hazard_index) * fp)  # 计算奖励
            else:  # Action affects 1 request  # 动作影响单个请求
                if is_user_request and action == 0:  # Accepted user  # 接受用户请求
                    return 0  # 返回0奖励
                elif is_user_request and action in [1, 2]:  # Rejected or blocked user  # 拒绝或阻止用户请求
                    return -2 / (max_load / self.load_threshold * self.hazard_index)  # 返回负奖励
                elif not is_user_request and action == 0:  # Accepted attacker  # 接受攻击者请求
                    return -(max_load / self.load_threshold * self.hazard_index)  # 返回负奖励
                elif not is_user_request and action in [1, 2]:  # Rejected or blocked attacker  # 拒绝或阻止攻击者请求
                    return max_load / self.load_threshold * self.hazard_index  # 返回正奖励
        else:  # 如果负载未超过阈值
            if action in [2, 3]:  # Block source or group  # 阻止源地址或组
                tp, fp = self.count_requests_in_buffer(group)  # 统计缓冲区中该组的真阳性和假阳性
                return tp - 2 * fp  # 返回简单奖励计算
            else:  # Action affects 1 request  # 动作影响单个请求
                if is_user_request and action == 0:  # Accepted user  # 接受用户请求
                    return 0  # 返回0奖励
                elif is_user_request and action in [1, 2]:  # Rejected or blocked user  # 拒绝或阻止用户请求
                    return -2  # 返回固定负奖励
                elif not is_user_request and action == 0:  # Accepted attacker  # 接受攻击者请求
                    return -1  # 返回固定负奖励
                elif not is_user_request and action in [1, 2]:  # Rejected or blocked attacker  # 拒绝或阻止攻击者请求
                    return 1  # 返回固定正奖励

    def count_requests_in_buffer(self, group):  # 统计缓冲区中指定组的请求数量
        # Count user and attacker requests in buffer  # 统计缓冲区中的用户和攻击者请求
        tp = 0  # True Positives (user requests)  # 真阳性（用户请求）
        fp = 0  # False Positives (attacker requests)  # 假阳性（攻击者请求）
        for state, ip, is_user_request in self.request_buffer:  # 遍历缓冲区中的所有请求
            if self.get_source_group(ip) == group:  # 如果IP地址属于指定组
                if is_user_request:  # 如果是用户请求
                    tp += 1  # 真阳性计数加1
                else:  # 如果是攻击者请求
                    fp += 1  # 假阳性计数加1
        return tp, fp  # 返回真阳性和假阳性计数

    def generate_random_ip(self):  # 生成随机IP地址的方法
        # Generate random IPv4 address, excluding special addresses  # 生成随机IPv4地址，排除特殊地址
        while True:  # 无限循环直到生成有效IP
            ip = str(ipaddress.IPv4Address(np.random.randint(0, 256**4)))  # 生成随机IP地址
            if ip not in ["127.0.0.1", "0.0.0.0"]:  # 如果不是特殊地址
                return ip  # 返回生成的IP地址

    def generate_random_ips(self, count):  # 生成指定数量的随机IP地址
        # Generate list of random IPv4 addresses  # 生成随机IPv4地址列表
        ips = set()  # 创建IP地址集合
        while len(ips) < count:  # 当IP地址数量不足时
            ips.add(self.generate_random_ip())  # 添加随机生成的IP地址
        return list(ips)  # 返回IP地址列表

    def get_source_group(self, source_ip):  # 获取源IP地址的组
        # Group by /24 subnet  # 按/24子网分组
        ip = ipaddress.IPv4Address(source_ip)  # 创建IP地址对象
        return str(ipaddress.IPv4Network(f"{ip}/24", strict=False).network_address)  # 返回网络地址（前24位）

    def generate_random_state(self):  # 生成随机状态的方法
        # Determine if request is user request  # 确定请求是否为用户请求
        is_user_request = np.random.rand() < self.user_request_prob  # 根据概率决定是否为用户请求
        
        # Generate IP address based on request type  # 根据请求类型生成IP地址
        if is_user_request:  # 如果是用户请求
            source_ip = np.random.choice(list(self.user_ips))  # 从用户IP池中随机选择
        else:  # 如果是攻击者请求
            while True:  # 无限循环直到生成非用户IP
                source_ip = self.generate_random_ip()  # 生成随机IP地址
                if source_ip not in self.user_ips:  # 如果IP不在用户IP池中
                    break  # 跳出循环
        
        # Generate message size  # 生成消息大小
        if is_user_request:  # 如果是用户请求
            message_size = np.random.choice(self.user_message_sizes)  # 从用户消息大小池中随机选择
        else:  # 如果是攻击者请求
            message_size = np.random.randint(100, 1000000)  # 随机生成100到1000000之间的消息大小
        
        # Group by /24 subnet  # 按/24子网分组
        group = self.get_source_group(source_ip)  # 获取源IP地址的组
        
        # Update traffic size by group  # 按组更新流量大小
        self.group_traffic[group] += message_size  # 累加该组的流量大小
        
        # Random CPU and memory load  # 随机CPU和内存负载
        cpu_load = np.random.uniform(0, 1)  # 生成0到1之间的随机CPU负载
        memory_load = np.random.uniform(0, 1)  # 生成0到1之间的随机内存负载
        
        # Generate source statistics  # 生成源地址统计信息
        source_stat = self.source_stats[source_ip]  # 获取源IP地址的统计信息
        if source_stat["last_time"] > 0:  # 如果之前有访问记录
            avg_time = np.mean(source_stat["intervals"]) if source_stat["intervals"] else 0  # 计算平均访问间隔
            avg_abs_deviation = np.mean(np.abs(np.array(source_stat["intervals"]) - avg_time)) if source_stat["intervals"] else 0  # 计算平均绝对偏差
        else:  # 如果之前没有访问记录
            avg_time = 0  # 平均时间设为0
            avg_abs_deviation = 0  # 平均绝对偏差设为0
        
        # Create filled state  # 创建完整状态
        state = np.array([  # 创建状态数组
            message_size,  # 消息大小
            self.group_traffic[group],  # 组流量大小
            avg_time,  # 平均访问间隔
            avg_abs_deviation,  # 平均绝对偏差
            source_stat["total_messages"],  # 总消息数量
            cpu_load,  # CPU负载
            memory_load,  # 内存负载
        ], dtype=np.float32)  # 数据类型为float32
        
        return state, source_ip, is_user_request  # 返回状态、源IP地址和用户请求标志

    def fill_buffer(self):  # 填充缓冲区的方法
        # Fill buffer to 100 elements  # 填充缓冲区到100个元素
        while len(self.request_buffer) < self.buffer_size:  # 当缓冲区大小不足时
            state, source_ip, is_user_request = self.generate_random_state()  # 生成随机状态
            self.request_buffer.append((state, source_ip, is_user_request))  # 将请求添加到缓冲区

    def add_request_to_buffer(self, state):  # 向缓冲区添加请求的方法
        # Determine request class: user or attacker  # 确定请求类型：用户或攻击者
        is_user_request = np.random.rand() < self.user_request_prob  # 根据概率决定是否为用户请求
        
        # Generate IP address based on request class  # 根据请求类型生成IP地址
        while True:  # 无限循环直到生成有效IP
            if is_user_request:  # 如果是用户请求
                source_ip = np.random.choice(list(self.user_ips))  # 从用户IP池中随机选择
            else:  # 如果是攻击者请求
                source_ip = self.generate_random_ip()  # 生成随机IP地址
                if source_ip not in self.user_ips:  # 如果IP不在用户IP池中
                    break  # 跳出循环
            
            # Check that IP address and its group are not blocked  # 检查IP地址及其组是否未被阻止
            if source_ip not in self.blocked_ips and self.get_source_group(source_ip) not in self.blocked_groups:  # 如果IP和组都未被阻止
                break  # 跳出循环
        
        # Add request to buffer  # 向缓冲区添加请求
        self.request_buffer.append((state, source_ip, is_user_request))  # 将请求信息添加到缓冲区

    def remove_blocked_requests(self):  # 移除被阻止请求的方法
        # Remove requests that belong to blocked IP addresses or groups  # 移除属于被阻止IP地址或组的请求
        self.request_buffer = deque(  # 重新创建双端队列
            (state, ip, is_user) for state, ip, is_user in self.request_buffer  # 遍历缓冲区中的所有请求
            if ip not in self.blocked_ips and self.get_source_group(ip) not in self.blocked_groups  # 只保留未被阻止的IP和组
        )

    def render(self, mode='human'):  # 渲染方法（用于可视化）
        pass  # 暂未实现