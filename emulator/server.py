# 导入套接字模块
import socket
# 导入日志模块
import logging
# 导入系统监控模块
import psutil
# 导入线程模块
import threading
# 导入Prometheus客户端
from prometheus_client import Gauge

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 创建CPU使用率监控指标
CPU_USAGE = Gauge('cpu_usage_percent', 'Current CPU usage in percent')
# 创建内存使用率监控指标
MEMORY_USAGE = Gauge('memory_usage_percent', 'Current memory usage in percent')

def update_system_metrics():
    """更新系统监控指标"""
    # 设置CPU使用率
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    # 设置内存使用率
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

def handle_client(conn, addr):
    """处理客户端连接的函数"""
    # 记录客户端连接信息
    logging.info(f"来自 {addr} 的连接")
    # 持续处理客户端数据
    while True:
        # 接收客户端数据
        data = conn.recv(1024)
        # 如果没有数据，退出循环
        if not data:
            break
        
        # 解码数据
        data = data.decode('utf-8')
        # 打印数据
        print(data)

        # 记录接收到的数据包
        logging.info(f"接收到数据包:\n{data}")

        # 处理获取状态请求
        if 'get' in data:
            if data.strip() == 'get state for features':
                # 获取特征状态
                state = f'{CPU_USAGE._value.get()} {MEMORY_USAGE._value.get()}'
            elif data.strip() == 'get server usage':
                # 获取服务器使用情况
                state = f'{psutil.cpu_percent(interval=1)} {psutil.virtual_memory().percent}'
                
            # 发送状态信息给客户端
            conn.sendall(str(state).encode('utf-8'))
        else:
            # 更新系统监控指标
            update_system_metrics()

    # 关闭连接
    conn.close()

def start(host='127.0.0.1', port=8080, memory_limit = 512 * 1024 * 1024):
    """启动服务器的主函数"""
    # 获取当前进程
    process = psutil.Process()
    
    # 检查操作系统类型
    if hasattr(process, 'rlimit'):  # 对于Unix系统
        # 设置内存限制
        process.rlimit(psutil.RLIMIT_AS, (memory_limit, memory_limit))
    
    # CPU亲和性限制适用于所有系统
    # 将进程绑定到CPU核心0
    process.cpu_affinity([0])

    # 创建TCP套接字
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 绑定到指定地址和端口
        s.bind((host, port))
        # 开始监听连接
        s.listen()
        # 记录服务器启动信息
        logging.info(f"服务器已启动在 {host}:{port}")

        # 持续接受客户端连接
        while True:
            # 接受客户端连接
            conn, addr = s.accept()
            # 处理客户端请求
            handle_client(conn, addr)


# 启动服务器
start()