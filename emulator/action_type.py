# 导入枚举模块
from enum import Enum

class Action_type(Enum):
    """动作类型枚举类"""
    # 单一目标动作
    SINGLE_TARGET_ACTION = 1
    # 多目标动作
    MULTILPE_TARGET_ACTION = 2

class Action(Enum):
    """具体动作枚举类"""
    # 服务器接收当前请求
    SERVER_RECIEVE_CURRENT = 0
    # 服务器丢弃当前请求
    SERVER_DROP_CURRENT = 1
    # 服务器阻止当前地址
    SERVER_BLOCK_CURRENT_ADDRESS = 2
    # 服务器阻止当前地址组
    SERVER_BLOCK_CURRENT_ADDRESS_GROUP = 3
    # 服务器丢弃相似请求（已注释）
    #SERVER_DROP_SIMULAR = 4
    