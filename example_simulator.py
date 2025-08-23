from simulator.sim_env import HTTPServerEnv  # 导入模拟器环境类
from agent.ddqn_agent import DoubleQAgent  # 导入双Q网络智能体类
import time  # 导入时间模块，用于计时
import numpy as np  # 导入数值计算库，用于数组操作和数学计算
import matplotlib.pyplot as plt  # 导入绘图库，用于可视化结果


if __name__=="__main__":  # 如果脚本直接运行（不是被导入）
    LEARN_EVERY = 4  # 每4步学习一次

    def train_agent(n_episodes=1200):  # 训练智能体的函数，默认1200回合
        print(f"Training a DDQN agent on {n_episodes} episodes.")  # 打印训练开始信息
        env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)  # 创建HTTP服务器环境实例
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

            if (i+1) % 10 == 0 and i > 0:  # 每10回合报告一次进度
                # Report expected time to finish the training  # 报告完成训练的预计时间
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1),  # 打印进度信息
                                                                                                                        (time.time() - start)/60,  # 已用时间（分钟）
                                                                                                                        n_episodes,  # 总回合数
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60,  # 预计总时间（分钟）
                                                                                                                        score,  # 当前回合得分
                                                                                                                        avg_score))  # 平均得分
                    
        return agent, scores  # 返回训练好的智能体和得分列表



    def test_agent(agent, n_episodes=200):  # 测试智能体的函数，默认200回合
        print(f"Testing a DDQN agent on {n_episodes} episodes.")  # 打印测试开始信息
        env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)  # 创建测试环境实例

        scores = []  # 初始化得分列表
        start = time.time()  # 记录测试开始时间
        for i in range(n_episodes):  # 循环测试指定回合数
            terminated = False  # 初始化终止标志
            score = 0  # 初始化回合得分
            state = env.reset()  # 重置环境并获取初始状态
            while not (terminated):  # 当回合未终止时继续
                action = agent.choose_action(state)  # 智能体选择动作
                new_state, reward, terminated, info = env.step(action)  # 执行动作并获取结果
                state = new_state  # 更新状态
                score += reward  # 累加奖励
                
            scores.append(score)  # 记录回合得分
            avg_score = np.mean(scores[max(0, i-100):(i+1)])  # 计算最近100回合的平均得分

            if (i+1) % 10 == 0 and i > 0:  # 每10回合报告一次进度
                # Report expected time to finish the training  # 报告完成测试的预计时间
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1),  # 打印进度信息
                                                                                                                        (time.time() - start)/60,  # 已用时间（分钟）
                                                                                                                        n_episodes,  # 总回合数
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60,  # 预计总时间（分钟）
                                                                                                                        score,  # 当前回合得分
                                                                                                                        avg_score))  # 平均得分
                    
        return scores  # 返回测试得分列表


    # Uncomment to train  # 取消注释以进行训练
    agent, ed_scores = train_agent(n_episodes=1200)  # 训练智能体1200回合
    
    # 保存训练好的模型  # 保存训练好的模型
    print("\n💾 保存训练好的模型...")  # 打印保存模型信息
    model_dir = "models"  # 模型目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    model_path = f"{model_dir}/ddqn_model_{timestamp}.pth"  # 构造模型文件路径
    
    # 保存模型（包含优化器状态，不包含经验回放缓冲区）  # 保存模型（包含优化器状态，不包含经验回放缓冲区）
    agent.save_model(model_path, save_optimizer=True, save_memory=False)  # 保存模型
    
    # 导出ONNX格式（用于生产部署）  # 导出ONNX格式（用于生产部署）
    onnx_path = f"{model_dir}/ddqn_model_{timestamp}.onnx"  # 构造ONNX文件路径
    print(f"\n🚀 导出ONNX模型...")  # 打印导出ONNX信息
    agent.export_onnx(onnx_path)  # 导出ONNX模型
    
    # 显示模型信息  # 显示模型信息
    print(f"\n📊 模型信息:")  # 打印模型信息标题
    model_info = agent.get_model_info()  # 获取模型信息
    for key, value in model_info.items():  # 遍历模型信息
        print(f"   {key}: {value}")  # 打印每个信息项
    
    # 测试模型  # 测试模型
    print(f"\n🧪 测试训练好的模型...")  # 打印测试开始信息
    agent.is_learning = False  # 关闭学习模式
    test_scores = test_agent(agent)  # 测试智能体
    
    # 可视化训练和测试结果  # 可视化训练和测试结果
    ed_indexes = np.arange(len(ed_scores)).tolist()  # 生成训练回合索引
    test_indexes = np.arange(len(test_scores)).tolist()  # 生成测试回合索引

    plt.figure(figsize=(12, 5))  # 创建图形，设置大小
    
    plt.subplot(1, 2, 1)  # 创建第一个子图
    plt.plot(ed_indexes, ed_scores)  # 绘制训练得分曲线
    plt.xlabel("训练回合")  # 设置x轴标签
    plt.ylabel("奖励")  # 设置y轴标签
    plt.title("训练期间智能体性能")  # 设置标题
    plt.grid(True)  # 显示网格
    
    plt.subplot(1, 2, 2)  # 创建第二个子图
    plt.plot(test_indexes, test_scores)  # 绘制测试得分曲线
    plt.xlabel("测试回合")  # 设置x轴标签
    plt.ylabel("奖励")  # 设置y轴标签
    plt.title("测试期间智能体性能")  # 设置标题
    plt.grid(True)  # 显示网格
    
    plt.tight_layout()  # 调整布局
    plt.show(block=True)  # 显示图形，阻塞模式
    
    print(f"\n🎉 训练完成！")  # 打印完成信息
    print(f"📁 模型文件: {model_path}")  # 打印模型文件路径
    print(f"🚀 ONNX模型: {onnx_path}")  # 打印ONNX模型路径
    print(f"📊 训练回合: {len(ed_scores)}")  # 打印训练回合数
    print(f"🧪 测试回合: {len(test_scores)}")  # 打印测试回合数
    print(f"📈 平均训练奖励: {np.mean(ed_scores):.2f}")  # 打印平均训练奖励
    print(f"📈 平均测试奖励: {np.mean(test_scores):.2f}")  # 打印平均测试奖励