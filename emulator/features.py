import numpy as np  # 导入数值计算库，用于数组操作和数学计算

from collections import defaultdict  # 导入默认字典，用于自动初始化字典值
    
class Features:  # 定义特征提取类
    def __init__(self):  # 初始化方法
        self.user_groups = defaultdict(list)  # 创建用户组字典，每个IP地址对应一个列表
        self.ip_blocks = defaultdict(int)  # 创建IP地址块字典，每个网络掩码对应一个整数计数器
    
    def get_netmask_from_ip(self, ip_address):  # 从IP地址获取网络掩码的方法
        ip_parts = ip_address.split('.')  # 将IP地址按点分割成部分
        return '.'.join(ip_parts[:-1])  # 返回前三个部分（网络部分），忽略主机部分

    def extract(self, message, interval, ip_address):  # 提取特征的方法
        if ':' in ip_address:  # 如果IP地址包含冒号（可能是IPv6或带端口的IPv4）
            ip_address = ip_address.split(':')[0]  # 提取冒号前的部分作为IP地址

        netmask = self.get_netmask_from_ip(ip_address=ip_address)  # 获取网络掩码

        self.user_groups[ip_address].append((netmask, interval, message))  # 将用户信息添加到用户组字典中

        bytes_m = len(message.encode('utf-8'))  # 计算当前消息的字节数
        self.ip_blocks[netmask] += bytes_m  # 累加该网络掩码下的总字节数
        bytes_b = self.ip_blocks[netmask]  # 获取该网络掩码下的总字节数

        user_intervals = np.array([item[1] for item in self.user_groups[ip_address]])  # 提取该IP地址的所有时间间隔

        ave = np.sum(user_intervals) / ((len(user_intervals) - 1) if len(user_intervals) -1  != 0 else 1)  # 计算平均时间间隔
        dev = (np.sum(user_intervals) - ave) / ((len(user_intervals) - 1) if len(user_intervals) -1  != 0 else 1)  # 计算时间间隔偏差
        num_m = len(self.user_groups[ip_address])  # 获取该IP地址的消息数量

        return np.array([bytes_m, bytes_b, ave, dev, num_m])  # 返回特征数组：当前消息字节数、总字节数、平均间隔、偏差、消息数量