from emulator import enviroment  # 导入仿真器环境模块
from agent.ddqn_agent import DoubleQAgent  # 导入双Q网络智能体类
import time  # 导入时间模块，用于计时
import numpy as np  # 导入数值计算库，用于数组操作和数学计算
import logging  # 导入日志模块，用于记录训练过程
from datetime import datetime  # 导入日期时间模块，用于生成时间戳

LEARN_EVERY = 4  # 每4步学习一次

def train_agent(n_episodes=2000):  # 训练智能体的函数，默认2000回合
    # Configure logging  # 配置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 生成时间戳格式：年月日_时分秒
    log_filename = f'training_log_{timestamp}.txt'  # 构造日志文件名
    
    # Create formatter for logs  # 创建日志格式化器
    formatter = logging.Formatter('%(asctime)s - %(message)s')  # 设置日志格式：时间 - 消息
    
    # Configure file handler  # 配置文件处理器
    file_handler = logging.FileHandler(log_filename)  # 创建文件处理器
    file_handler.setFormatter(formatter)  # 设置格式化器
    
    # Configure logger  # 配置日志记录器
    logger = logging.getLogger('training')  # 创建名为'training'的日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    logger.addHandler(file_handler)  # 添加文件处理器

    print(f"Starting training DDQN agent on {n_episodes} episodes.")  # 打印训练开始信息
    env = enviroment.TrafficEnv(mode="emulator")  # 创建仿真器环境实例

    agent = DoubleQAgent(observation_space_shape=env.observation_space.shape[0], action_space_n=env.action_space.n,  # 创建双Q网络智能体实例
                             gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)  # 设置超参数：折扣因子、探索率、学习率等
            
    scores = []  # 初始化得分列表
    eps_history = []  # 初始化探索率历史列表
    start = time.time()  # 记录训练开始时间
    
    for i in range(n_episodes):  # 循环训练指定回合数
        terminated = False  # 初始化终止标志
        score = 0  # 初始化回合得分
        state = env.reset()  # 重置环境并获取初始状态
        steps = 0  # 初始化步数计数器
        while not (terminated):  # 当回合未终止时继续
            env.get_state()  # 获取环境状态
            action = agent.choose_action(state)  # 智能体选择动作
            new_state, reward, terminated, info = env.step(action)  # 执行动作并获取结果
            agent.save(state, action, reward, new_state, terminated)  # 保存经验到缓冲区
            state = new_state  # 更新状态
            if steps > 0 and steps % LEARN_EVERY == 0:  # 如果步数大于0且是学习间隔的倍数
                agent.learn()  # 智能体学习
            steps += 1  # 步数加1
            score += reward  # 累加奖励
                
        eps_history.append(agent.epsilon)  # 记录当前探索率
        scores.append(score)  # 记录回合得分
        avg_score = np.mean(scores[max(0, i-100):(i+1)])  # 计算最近100回合的平均得分

        log_message = 'Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format(  # 构造日志消息
            (i+1),  # 当前回合数
            (time.time() - start)/60,  # 已用时间（分钟）
            n_episodes,  # 总回合数
            (((time.time() - start)/(i+1))*n_episodes)/60,  # 预计总时间（分钟）
            score,  # 当前回合得分
            avg_score  # 平均得分
        )
        
        logger.info(log_message)  # 记录日志信息
                    
    logger.info("Training completed!")  # 记录训练完成信息
    return agent, scores  # 返回训练好的智能体和得分列表

def main():  # 主函数
    train_agent()  # 调用训练函数

if __name__ == '__main__':  # 如果脚本直接运行（不是被导入）
    main()  # 调用主函数