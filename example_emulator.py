# 导入环境模块
from emulator import enviroment
# 导入双深度Q网络智能体
from agent.ddqn_agent import DoubleQAgent
# 导入时间模块
import time
# 导入数值计算库
import numpy as np
# 导入日志模块
import logging
# 导入日期时间模块
from datetime import datetime

# 定义学习频率，每4步学习一次
LEARN_EVERY = 4

def train_agent(n_episodes=2000):
    # 设置日志记录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_log_{timestamp}.txt'
    
    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # 设置文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # 设置日志记录器
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    print(f"Starting training DDQN agent on {n_episodes} episodes.")
    # 创建交通环境实例，模式为模拟器
    env = enviroment.TrafficEnv(mode="emulator")

    # 创建双深度Q网络智能体实例
    agent = DoubleQAgent(observation_space_shape=env.observation_space.shape[0], action_space_n=env.action_space.n, 
                             gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
            
    # 初始化分数历史列表
    scores = []
    # 初始化epsilon历史列表
    eps_history = []
    # 记录开始时间
    start = time.time()
    
    # 开始训练循环
    for i in range(n_episodes):
        # 初始化终止标志
        terminated = False
        # 初始化分数
        score = 0
        # 重置环境并获取初始状态
        state = env.reset()
        # 初始化步数
        steps = 0
        # 当环境未终止时继续执行
        while not (terminated):
            # 获取当前状态
            env.get_state()
            # 智能体选择动作
            action = agent.choose_action(state)
            # 执行动作并获取新状态、奖励、终止标志和信息
            new_state, reward, terminated, info = env.step(action)
            # 保存经验到记忆缓冲区
            agent.save(state, action, reward, new_state, terminated)
            # 更新状态
            state = new_state
            # 如果步数大于0且是学习频率的倍数，则进行学习
            if steps > 0 and steps % LEARN_EVERY == 0:
                agent.learn()
            # 步数加1
            steps += 1
            # 累加奖励
            score += reward
                
        # 记录epsilon值
        eps_history.append(agent.epsilon)
        # 记录分数
        scores.append(score)
        # 计算最近100个episode的平均分数
        avg_score = np.mean(scores[max(0, i-100):(i+1)])

        # 构建日志消息
        log_message = 'Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format(
            (i+1), 
            (time.time() - start)/60, 
            n_episodes, 
            (((time.time() - start)/(i+1))*n_episodes)/60, 
            score, 
            avg_score
        )
        
        # 记录日志
        logger.info(log_message)
                    
    # 记录训练完成信息
    logger.info("Training completed!")
    # 返回训练好的智能体和分数历史
    return agent, scores

def main():
    # 执行训练函数
    train_agent()

# 如果直接运行此脚本，则执行main函数
if __name__ == '__main__':
    main()