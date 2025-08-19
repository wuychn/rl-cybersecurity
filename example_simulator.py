from simulator.sim_env import HTTPServerEnv
from agent.ddqn_agent import DoubleQAgent
import time
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    LEARN_EVERY = 4

    def train_agent(n_episodes=1200):
        print(f"Training a DDQN agent on {n_episodes} episodes.")
        env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)
        agent = DoubleQAgent(observation_space_shape=env.observation_space.shape[0], action_space_n=env.action_space.n, 
                             gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
            
        scores = []
        eps_history = []
        start = time.time()
        for i in range(n_episodes):
            terminated = False
            score = 0
            state = env.reset()
            steps = 0
            while not (terminated):
                action = agent.choose_action(state)
                new_state, reward, terminated, info = env.step(action)
                agent.save(state, action, reward, new_state, terminated)
                state = new_state
                if steps > 0 and steps % LEARN_EVERY == 0:
                    agent.learn()
                steps += 1
                score += reward
                
            eps_history.append(agent.epsilon)
            scores.append(score)
            avg_score = np.mean(scores[max(0, i-100):(i+1)])

            if (i+1) % 10 == 0 and i > 0:
                # Report expected time to finish the training
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                        (time.time() - start)/60, 
                                                                                                                        n_episodes, 
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                        score, 
                                                                                                                        avg_score))
                    
        return agent, scores



    def test_agent(agent, n_episodes=200):
        print(f"Testing a DDQN agent on {n_episodes} episodes.")
        env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)

        scores = []
        start = time.time()
        for i in range(n_episodes):
            terminated = False
            score = 0
            state = env.reset()
            while not (terminated):
                action = agent.choose_action(state)
                new_state, reward, terminated, info = env.step(action)
                state = new_state
                score += reward
                
            scores.append(score)
            avg_score = np.mean(scores[max(0, i-100):(i+1)])

            if (i+1) % 10 == 0 and i > 0:
                # Report expected time to finish the training
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                        (time.time() - start)/60, 
                                                                                                                        n_episodes, 
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                        score, 
                                                                                                                        avg_score))
                    
        return scores


    # Uncomment to train
    agent, ed_scores = train_agent(n_episodes=1200)
    
    # 保存训练好的模型
    print("\n💾 保存训练好的模型...")
    model_dir = "models"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = f"{model_dir}/ddqn_model_{timestamp}.pth"
    
    # 保存模型（包含优化器状态，不包含经验回放缓冲区）
    agent.save_model(model_path, save_optimizer=True, save_memory=False)
    
    # 导出ONNX格式（用于生产部署）
    onnx_path = f"{model_dir}/ddqn_model_{timestamp}.onnx"
    print(f"\n🚀 导出ONNX模型...")
    agent.export_onnx(onnx_path)
    
    # 显示模型信息
    print(f"\n📊 模型信息:")
    model_info = agent.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # 测试模型
    print(f"\n🧪 测试训练好的模型...")
    agent.is_learning = False
    test_scores = test_agent(agent)
    
    # 可视化训练和测试结果
    ed_indexes = np.arange(len(ed_scores)).tolist()
    test_indexes = np.arange(len(test_scores)).tolist()

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ed_indexes, ed_scores)
    plt.xlabel("训练回合")
    plt.ylabel("奖励")
    plt.title("训练期间智能体性能")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_indexes, test_scores)
    plt.xlabel("测试回合")
    plt.ylabel("奖励")
    plt.title("测试期间智能体性能")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"\n🎉 训练完成！")
    print(f"📁 模型文件: {model_path}")
    print(f"🚀 ONNX模型: {onnx_path}")
    print(f"📊 训练回合: {len(ed_scores)}")
    print(f"🧪 测试回合: {len(test_scores)}")
    print(f"📈 平均训练奖励: {np.mean(ed_scores):.2f}")
    print(f"📈 平均测试奖励: {np.mean(test_scores):.2f}")