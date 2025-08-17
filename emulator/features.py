# 导入数值计算库
import numpy as np

# 导入默认字典
from collections import defaultdict
    
class Features:
    """特征提取类，用于从网络消息中提取特征"""
    def __init__(self):
        # 用户组字典，存储每个IP地址的消息列表
        self.user_groups = defaultdict(list)
        # IP块字典，存储每个网段的字节数
        self.ip_blocks = defaultdict(int)
    
    def get_netmask_from_ip(self, ip_address):
        """从IP地址中提取网段"""
        # 分割IP地址
        ip_parts = ip_address.split('.')
        # 返回前三个部分作为网段
        return '.'.join(ip_parts[:-1])

    def extract(self, message, interval, ip_address):
        """提取特征的方法"""
        # 如果IP地址包含端口号，则只取IP部分
        if ':' in ip_address:
            ip_address = ip_address.split(':')[0]

        # 获取网段
        netmask = self.get_netmask_from_ip(ip_address=ip_address)

        # 将消息信息添加到用户组中
        self.user_groups[ip_address].append((netmask, interval, message))

        # 计算当前消息的字节数
        bytes_m = len(message.encode('utf-8'))
        # 累加该网段的总字节数
        self.ip_blocks[netmask] += bytes_m
        # 获取该网段的总字节数
        bytes_b = self.ip_blocks[netmask]

        # 获取该用户的所有时间间隔
        user_intervals = np.array([item[1] for item in self.user_groups[ip_address]])

        # 计算平均时间间隔
        ave = np.sum(user_intervals) / ((len(user_intervals) - 1) if len(user_intervals) -1  != 0 else 1)
        # 计算时间间隔偏差
        dev = (np.sum(user_intervals) - ave) / ((len(user_intervals) - 1) if len(user_intervals) -1  != 0 else 1)
        # 计算该用户的消息数量
        num_m = len(self.user_groups[ip_address])

        # 返回特征数组：[当前消息字节数, 网段总字节数, 平均时间间隔, 时间间隔偏差, 消息数量]
        return np.array([bytes_m, bytes_b, ave, dev, num_m])