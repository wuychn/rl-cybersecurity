# 导入套接字模块
import socket
# 导入字符串模块
import string
# 导入随机数模块
import random
# 导入日志模块
import logging
# 导入时间模块
import time

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 全局用户服务器变量
_user_server = None

def send_messages(agent_host, agent_port, host, port, min_intv, max_intv) -> str:
    """发送消息的函数"""
    global _user_server
    # 定义可用字符集（字母和数字）
    symbols = string.ascii_letters+string.digits
    
    # 无限循环发送消息
    while True:
        try:
            # 创建TCP套接字
            _user_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置套接字选项，允许地址重用
            _user_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 绑定到指定地址和端口
            _user_server.bind((host, port))
            # 连接到智能体服务器
            _user_server.connect((agent_host,agent_port))
            
            # 持续发送消息
            while True:
                # 随机决定是否为用户消息（35%概率）
                is_user = random.random() < 0.35

                if is_user:
                    # 生成用户IP地址（1-200范围）
                    ip_address = '72.166.25.' + str(random.randint(1,200))
                    # 生成随机消息（1-30字符）
                    message = ''.join(random.choice(symbols) for _ in range(random.randint(1,30)))
                    # 设置时间间隔为2秒
                    interval = 2
                else:
                    # 生成机器人IP地址（200-255范围）
                    ip_address = '72.166.25.' + str(random.randint(200,255))
                    # 生成随机消息（1-5字符）
                    message = ''.join(random.choice(symbols) for _ in range(random.randint(1,5)))
                    # 设置时间间隔为1秒
                    interval = 1
                # 注释掉的时间间隔随机化
                # interval = random.randint(min_intv,max_intv)
                # 构建完整消息格式：消息@@IP地址@@是否用户
                message = f'{message}@@{ip_address}@@{is_user}'
                # 发送消息
                _user_server.sendall(message.encode('utf-8'))
                # 记录日志
                logging.info(f"{'用户' if is_user else '机器人'} 发送消息到 {agent_host}:{agent_port}")
                # 等待指定时间间隔
                time.sleep(interval)
                
        except (ConnectionAbortedError, ConnectionResetError, ConnectionRefusedError) as e:
            # 处理连接错误
            logging.warning(f"连接中断: {e}. 正在重新连接...")
            if '_user_server' in locals():
                _user_server.close()

        except Exception as e:
            # 处理其他异常
            logging.error(f"意外错误: {e}")
            if '_user_server' in locals():
                _user_server.close()
            raise

def start(host='127.0.0.1', port=8070, agent_host='127.0.0.1', agent_port=8090, 
          min_intv=3, max_intv=10):
    """启动消息生成器的主函数"""
    # 调用发送消息函数
    send_messages(agent_host=agent_host, agent_port=agent_port, host=host, 
                 port=port, min_intv=min_intv, max_intv=max_intv)



