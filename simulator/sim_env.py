import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ipaddress
from collections import deque, defaultdict

class HTTPServerEnv(gym.Env):
    def __init__(self, buffer_size=100, user_ips=None, load_threshold=0.8, hazard_index=1, user_message_sizes=None, user_request_prob=0.2):
        super(HTTPServerEnv, self).__init__()
        
        # Buffer and user address parameters
        self.buffer_size = buffer_size
        self.user_ips = set(user_ips) if user_ips else set(self.generate_random_ips(250))
        
        # Load and hazard index parameters
        self.load_threshold = load_threshold
        self.hazard_index = hazard_index
        
        # Pool of fixed message sizes for user addresses
        self.user_message_sizes = user_message_sizes if user_message_sizes else [100, 200, 300, 400, 500]
        
        # Probability of adding a user request
        self.user_request_prob = user_request_prob
        
        # Define state space (7 elements)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000000, 1000000, 1000, 1000, 10000, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Discrete(4)  # 0 - accept, 1 - reject, 2 - block source, 3 - block group
        
        # Simulation parameters
        self.max_messages = 700
        self.current_message_count = 0
        self.blocked_ips = set()
        self.blocked_groups = set()
        
        # Source statistics
        self.source_stats = defaultdict(lambda: {
            "last_time": 0,
            "intervals": [],
            "total_messages": 0,
        })
        
        # Traffic size by groups
        self.group_traffic = defaultdict(int)
        
        # Request buffer (stores state, source_ip, is_user_request)
        self.request_buffer = deque(maxlen=self.buffer_size)

    def reset(self):
        # Reset environment to initial state
        self.current_message_count = 0
        self.blocked_ips = set()
        self.blocked_groups = set()
        self.source_stats = defaultdict(lambda: {
            "last_time": 0,
            "intervals": [],
            "total_messages": 0,
        })
        self.group_traffic = defaultdict(int)
        self.request_buffer.clear()
        self.fill_buffer()  # Primary buffer filling
        return self.request_buffer[0][0]

    def step(self, action):
        # Simulate request processing
        done = False
        info = {}
        
        # Get current request from buffer with removal
        state, source_ip, is_user_request = self.request_buffer.popleft()
        
        # Update source statistics
        current_time = self.current_message_count
        source_stat = self.source_stats[source_ip]
        if source_stat["last_time"] > 0:
            interval = current_time - source_stat["last_time"]
            source_stat["intervals"].append(interval)
        source_stat["last_time"] = current_time
        source_stat["total_messages"] += 1
        
        # Calculate average time and deviation
        if len(source_stat["intervals"]) > 0:
            avg_time = np.mean(source_stat["intervals"])
            avg_abs_deviation = np.mean(np.abs(np.array(source_stat["intervals"]) - avg_time))
        else:
            avg_time = 0
            avg_abs_deviation = 0
        
        # Update traffic size by group
        group = self.get_source_group(source_ip)
        if is_user_request:
            message_size = np.random.choice(self.user_message_sizes)
        else:
            message_size = np.random.randint(100, 1000000)
        self.group_traffic[group] += message_size
        
        # Random CPU and memory load
        cpu_load = np.random.uniform(0, 1)
        memory_load = np.random.uniform(0, 1)
        
        # Create new state using generate_random_state
        new_state, _, _ = self.generate_random_state()
        
        # Calculate reward
        reward = self.get_reward(action, is_user_request, source_ip, group, cpu_load, memory_load)
        
        # Handle actions: block IP address or group
        if action == 2:  # Block source
            self.blocked_ips.add(source_ip)
        elif action == 3:  # Block group
            self.blocked_groups.add(group)
        
        # Remove blocked requests from buffer
        self.remove_blocked_requests()
        
        # Add new request to buffer
        self.add_request_to_buffer(new_state)
        
        # Fill buffer to 100 elements if it emptied after removing blocked requests
        while len(self.request_buffer) < self.buffer_size:
            self.add_request_to_buffer(self.generate_random_state()[0])
        
        # Check completion condition
        self.current_message_count += 1
        if self.current_message_count >= self.max_messages:
            done = True
        
        # Return new state (zeroth buffer element without removal)
        next_state = self.request_buffer[0][0]
        return next_state, reward, done, info

    def get_reward(self, action, is_user_request, source_ip, group, cpu_load, memory_load):
        # Calculate reward based on action
        max_load = max(cpu_load, memory_load)
        
        if max_load > self.load_threshold:
            if action in [2, 3]:  # Block source or group
                tp, fp = self.count_requests_in_buffer(group)
                return (max_load / self.load_threshold * self.hazard_index) * tp - (2 / (max_load / self.load_threshold * self.hazard_index) * fp)
            else:  # Action affects 1 request
                if is_user_request and action == 0:  # Accepted user
                    return 0
                elif is_user_request and action in [1, 2]:  # Rejected or blocked user
                    return -2 / (max_load / self.load_threshold * self.hazard_index)
                elif not is_user_request and action == 0:  # Accepted attacker
                    return -(max_load / self.load_threshold * self.hazard_index)
                elif not is_user_request and action in [1, 2]:  # Rejected or blocked attacker
                    return max_load / self.load_threshold * self.hazard_index
        else:
            if action in [2, 3]:  # Block source or group
                tp, fp = self.count_requests_in_buffer(group)
                return tp - 2 * fp
            else:  # Action affects 1 request
                if is_user_request and action == 0:  # Accepted user
                    return 0
                elif is_user_request and action in [1, 2]:  # Rejected or blocked user
                    return -2
                elif not is_user_request and action == 0:  # Accepted attacker
                    return -1
                elif not is_user_request and action in [1, 2]:  # Rejected or blocked attacker
                    return 1

    def count_requests_in_buffer(self, group):
        # Count user and attacker requests in buffer
        tp = 0  # True Positives (user requests)
        fp = 0  # False Positives (attacker requests)
        for state, ip, is_user_request in self.request_buffer:
            if self.get_source_group(ip) == group:
                if is_user_request:
                    tp += 1
                else:
                    fp += 1
        return tp, fp

    def generate_random_ip(self):
        # Generate random IPv4 address, excluding special addresses
        while True:
            ip = str(ipaddress.IPv4Address(np.random.randint(0, 256**4)))
            if ip not in ["127.0.0.1", "0.0.0.0"]:
                return ip

    def generate_random_ips(self, count):
        # Generate list of random IPv4 addresses
        ips = set()
        while len(ips) < count:
            ips.add(self.generate_random_ip())
        return list(ips)

    def get_source_group(self, source_ip):
        # Group by /24 subnet
        ip = ipaddress.IPv4Address(source_ip)
        return str(ipaddress.IPv4Network(f"{ip}/24", strict=False).network_address)

    def generate_random_state(self):
        # Determine if request is user request
        is_user_request = np.random.rand() < self.user_request_prob
        
        # Generate IP address based on request type
        if is_user_request:
            source_ip = np.random.choice(list(self.user_ips))
        else:
            while True:
                source_ip = self.generate_random_ip()
                if source_ip not in self.user_ips:
                    break
        
        # Generate message size
        if is_user_request:
            message_size = np.random.choice(self.user_message_sizes)
        else:
            message_size = np.random.randint(100, 1000000)
        
        # Group by /24 subnet
        group = self.get_source_group(source_ip)
        
        # Update traffic size by group
        self.group_traffic[group] += message_size
        
        # Random CPU and memory load
        cpu_load = np.random.uniform(0, 1)
        memory_load = np.random.uniform(0, 1)
        
        # Generate source statistics
        source_stat = self.source_stats[source_ip]
        if source_stat["last_time"] > 0:
            avg_time = np.mean(source_stat["intervals"]) if source_stat["intervals"] else 0
            avg_abs_deviation = np.mean(np.abs(np.array(source_stat["intervals"]) - avg_time)) if source_stat["intervals"] else 0
        else:
            avg_time = 0
            avg_abs_deviation = 0
        
        # Create filled state
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
        # Fill buffer to 100 elements
        while len(self.request_buffer) < self.buffer_size:
            state, source_ip, is_user_request = self.generate_random_state()
            self.request_buffer.append((state, source_ip, is_user_request))

    def add_request_to_buffer(self, state):
        # Determine request class: user or attacker
        is_user_request = np.random.rand() < self.user_request_prob
        
        # Generate IP address based on request class
        while True:
            if is_user_request:
                source_ip = np.random.choice(list(self.user_ips))
            else:
                source_ip = self.generate_random_ip()
                if source_ip not in self.user_ips:
                    break
            
            # Check that IP address and its group are not blocked
            if source_ip not in self.blocked_ips and self.get_source_group(source_ip) not in self.blocked_groups:
                break
        
        # Add request to buffer
        self.request_buffer.append((state, source_ip, is_user_request))

    def remove_blocked_requests(self):
        # Remove requests that belong to blocked IP addresses or groups
        self.request_buffer = deque(
            (state, ip, is_user) for state, ip, is_user in self.request_buffer
            if ip not in self.blocked_ips and self.get_source_group(ip) not in self.blocked_groups
        )

    def render(self, mode='human'):
        pass