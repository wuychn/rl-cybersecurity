# 导入HTTP服务器环境
from simulator.sim_env import HTTPServerEnv
# 导入双深度Q网络智能体
from agent.ddqn_agent import DoubleQAgent
# 导入时间模块
import time
# 导入数值计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt


# 如果直接运行此脚本
if __name__=="__main__":
    # 定义学习频率，每4步学习一次
    LEARN_EVERY = 4

    def train_agent(n_episodes=1200):
        # 打印训练信息
        print(f"Training a DDQN agent on {n_episodes} episodes.")
        # 创建HTTP服务器环境实例
        env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)
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

            # 每10个episode报告一次进度
            if (i+1) % 10 == 0 and i > 0:
                # 报告预计完成训练的时间
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                        (time.time() - start)/60, 
                                                                                                                        n_episodes, 
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                        score, 
                                                                                                                        avg_score))
                    
        # 返回训练好的智能体和分数历史
        return agent, scores



    def test_agent(agent, n_episodes=200):
        # 打印测试信息
        print(f"Testing a DDQN agent on {n_episodes} episodes.")
        # 创建HTTP服务器环境实例
        env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)

        # 初始化分数列表
        scores = []
        # 记录开始时间
        start = time.time()
        # 开始测试循环
        for i in range(n_episodes):
            # 初始化终止标志
            terminated = False
            # 初始化分数
            score = 0
            # 重置环境并获取初始状态
            state = env.reset()
            # 当环境未终止时继续执行
            while not (terminated):
                # 智能体选择动作
                action = agent.choose_action(state)
                # 执行动作并获取新状态、奖励、终止标志和信息
                new_state, reward, terminated, info = env.step(action)
                # 更新状态
                state = new_state
                # 累加奖励
                score += reward
                
            # 记录分数
            scores.append(score)
            # 计算最近100个episode的平均分数
            avg_score = np.mean(scores[max(0, i-100):(i+1)])

            # 每10个episode报告一次进度
            if (i+1) % 10 == 0 and i > 0:
                # 报告预计完成测试的时间
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                        (time.time() - start)/60, 
                                                                                                                        n_episodes, 
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                        score, 
                                                                                                                        avg_score))
                    
        # 返回测试分数
        return scores


    # 取消注释以开始训练
    # 训练智能体
    agent, ed_scores = train_agent(n_episodes=1200)
    # 关闭学习模式
    agent.is_learning = False
    # 测试智能体
    test_scores = test_agent(agent)
    # 创建训练episode索引列表
    ed_indexes = np.arange(len(ed_scores)).tolist()
    # 创建测试episode索引列表
    test_indexes = np.arange(len(test_scores)).tolist()

    # 绘制训练结果
    plt.plot(ed_indexes, ed_scores)
    plt.xlabel("训练回合")
    plt.ylabel("奖励")
    plt.title("智能体在训练过程中的表现")
    plt.show(block=True)

    # 绘制测试结果
    plt.plot(test_indexes, test_scores)
    plt.xlabel("测试回合")
    plt.ylabel("奖励")
    plt.title("智能体在测试过程中的表现")
    plt.show(block=True)