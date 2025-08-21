import socket  # 导入套接字模块，用于网络通信
import string  # 导入字符串模块，用于生成随机字符串
import random  # 导入随机数模块，用于生成随机数据
import logging  # 导入日志模块，用于记录程序运行信息
import time  # 导入时间模块，用于时间延迟

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')  # 配置日志格式和级别

_user_server = None  # 全局变量，存储用户服务器套接字对象

def send_messages(agent_host, agent_port, host, port, min_intv, max_intv) -> str:  # 发送消息的主函数
    global _user_server  # 声明使用全局变量
    symbols = string.ascii_letters+string.digits  # 定义可用字符集（字母和数字）
    
    while True:
        try:
            _user_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _user_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            _user_server.bind((host, port))
            _user_server.connect((agent_host,agent_port))
            
            while True:
                is_user = random.random() < 0.35

                if is_user:
                    ip_address = '72.166.25.' + str(random.randint(1,200))
                    message = ''.join(random.choice(symbols) for _ in range(random.randint(1,30)))
                    interval = 2
                else:
                    ip_address = '72.166.25.' + str(random.randint(200,255))
                    message = ''.join(random.choice(symbols) for _ in range(random.randint(1,5)))
                    interval = 1
                # interval = random.randint(min_intv,max_intv)  # 注释掉的随机间隔代码
                message = f'{message}@@{ip_address}@@{is_user}'  # 构造消息格式：消息内容@@IP地址@@用户类型
                _user_server.sendall(message.encode('utf-8'))  # 发送消息到智能体服务器
                logging.info(f"{'User' if is_user else 'Bot'} sent message to {agent_host}:{agent_port}")  # 记录发送信息
                time.sleep(interval)  # 等待指定间隔时间
                
        except (ConnectionAbortedError, ConnectionResetError, ConnectionRefusedError) as e:  # 捕获连接相关异常
            logging.warning(f"Connection interrupted: {e}. Reconnecting...")  # 记录连接中断警告
            if '_user_server' in locals():  # 如果套接字对象存在
                _user_server.close()  # 关闭套接字

        except Exception as e:  # 捕获其他异常
            logging.error(f"Unexpected error: {e}")  # 记录意外错误
            if '_user_server' in locals():  # 如果套接字对象存在
                _user_server.close()  # 关闭套接字
            raise  # 重新抛出异常

def start(host='127.0.0.1', port=8070, agent_host='127.0.0.1', agent_port=8090,  # 启动函数的默认参数
          min_intv=3, max_intv=10):  # 最小和最大间隔时间
    send_messages(agent_host=agent_host, agent_port=agent_port, host=host,  # 调用发送消息函数
                 port=port, min_intv=min_intv, max_intv=max_intv)  # 传递所有参数



