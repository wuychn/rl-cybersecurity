import socket  # 导入套接字模块，用于网络通信
import logging  # 导入日志模块，用于记录程序运行信息
import psutil  # 导入系统监控模块，用于获取CPU和内存使用率
import threading  # 导入线程模块，用于多线程处理
from prometheus_client import Gauge  # 导入Prometheus客户端，用于监控指标

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')  # 配置日志格式和级别

CPU_USAGE = Gauge('cpu_usage_percent', 'Current CPU usage in percent')  # 创建CPU使用率监控指标
MEMORY_USAGE = Gauge('memory_usage_percent', 'Current memory usage in percent')  # 创建内存使用率监控指标

def update_system_metrics():  # 更新系统指标的函数
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  # 更新CPU使用率指标
    MEMORY_USAGE.set(psutil.virtual_memory().percent)  # 更新内存使用率指标

def handle_client(conn, addr):  # 处理客户端连接的函数
    logging.info(f"Connection from {addr}")  # 记录客户端连接信息
    while True:  # 无限循环处理客户端请求
        data = conn.recv(1024)  # 接收客户端数据
        if not data:  # 如果没有接收到数据
            break  # 跳出循环
        
        data = data.decode('utf-8')  # 解码数据
        print(data)  # 打印接收到的数据

        logging.info(f"Received packet:\n{data}")  # 记录接收到的数据包

        if 'get' in data:  # 如果数据包含'get'关键字
            if data.strip() == 'get state for features':  # 如果是获取特征状态的请求
                state = f'{CPU_USAGE._value.get()} {MEMORY_USAGE._value.get()}'  # 返回监控指标值
            elif data.strip() == 'get server usage':  # 如果是获取服务器使用率的请求
                state = f'{psutil.cpu_percent(interval=1)} {psutil.virtual_memory().percent}'  # 返回实时系统指标
                
            conn.sendall(str(state).encode('utf-8'))  # 发送响应给客户端
        else:  # 如果不是获取请求
            update_system_metrics()  # 更新系统监控指标

    conn.close()  # 关闭客户端连接

def start(host='127.0.0.1', port=8080, memory_limit = 512 * 1024 * 1024):  # 启动服务器的函数
    process = psutil.Process()  # 获取当前进程对象
    
    # Check operating system  # 检查操作系统
    if hasattr(process, 'rlimit'):  # For Unix systems  # 如果是Unix系统
        process.rlimit(psutil.RLIMIT_AS, (memory_limit, memory_limit))  # 设置内存限制
    
    # CPU limitation works on all systems  # CPU限制在所有系统上都有效
    process.cpu_affinity([0])  # 限制进程只能使用第0个CPU核心

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建TCP套接字
        s.bind((host, port))  # 绑定地址和端口
        s.listen()  # 开始监听连接
        logging.info(f"Server started on {host}:{port}")  # 记录服务器启动信息

        while True:  # 无限循环接受连接
            conn, addr = s.accept()  # 接受客户端连接
            handle_client(conn, addr)  # 处理客户端请求


start()  # 启动服务器