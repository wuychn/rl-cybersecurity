from enum import Enum  # 导入枚举模块，用于定义常量枚举类型

class Action_type(Enum):  # 定义动作类型枚举类
    SINGLE_TARGET_ACTION = 1  # 单目标动作类型，只影响单个IP地址
    MULTILPE_TARGET_ACTION = 2  # 多目标动作类型，影响整个IP地址组

class Action(Enum):  # 定义具体动作枚举类
    SERVER_RECIEVE_CURRENT = 0  # 服务器接收当前请求
    SERVER_DROP_CURRENT = 1  # 服务器丢弃当前请求
    SERVER_BLOCK_CURRENT_ADDRESS = 2  # 服务器阻止当前IP地址
    SERVER_BLOCK_CURRENT_ADDRESS_GROUP = 3  # 服务器阻止当前IP地址组
    #SERVER_DROP_SIMULAR = 4  # 服务器丢弃相似请求（已注释）
    