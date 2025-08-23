#!/usr/bin/env python3
"""
模型加载和使用示例脚本
演示如何加载训练好的DDQN模型并进行推理
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append('.')

from agent.ddqn_agent import DoubleQAgent
from simulator.sim_env import HTTPServerEnv

def load_trained_model(model_path):
    """加载训练好的模型"""
    print(f"🔄 正在加载模型: {model_path}")
    
    # 创建智能体实例
    agent = DoubleQAgent(observation_space_shape=7, action_space_n=4)
    
    try:
        # 加载模型
        agent.load_model(model_path, load_optimizer=False, load_memory=False)
        print("✅ 模型加载成功！")
        return agent
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_loaded_model(agent, n_episodes=100):
    """测试加载的模型"""
    print(f"\n🧪 测试加载的模型 ({n_episodes} 回合)...")
    
    env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)
    scores = []
    actions_taken = []
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_score = 0
        episode_actions = []
        terminated = False
        steps = 0
        
        while not terminated and steps < 1000:  # 限制最大步数
            # 使用加载的模型选择动作
            action = agent.choose_action(state)
            episode_actions.append(action)
            
            # 执行动作
            new_state, reward, terminated, info = env.step(action)
            episode_score += reward
            state = new_state
            steps += 1
        
        scores.append(episode_score)
        actions_taken.append(episode_actions)
        
        if (episode + 1) % 20 == 0:
            elapsed = time.time() - start_time
            avg_score = np.mean(scores[-20:])
            print(f"回合 {episode + 1}/{n_episodes} - 得分: {episode_score:.2f}, 平均得分: {avg_score:.2f}, 用时: {elapsed:.1f}s")
    
    return scores, actions_taken

def analyze_model_performance(scores, actions_taken):
    """分析模型性能"""
    print(f"\n📊 模型性能分析:")
    print(f"   - 总回合数: {len(scores)}")
    print(f"   - 平均得分: {np.mean(scores):.2f}")
    print(f"   - 最高得分: {np.max(scores):.2f}")
    print(f"   - 最低得分: {np.min(scores):.2f}")
    print(f"   - 得分标准差: {np.std(scores):.2f}")
    
    # 分析动作分布
    all_actions = [action for episode_actions in actions_taken for action in episode_actions]
    action_counts = np.bincount(all_actions, minlength=4)
    action_names = ['接受', '拒绝', '阻止源', '阻止组']
    
    print(f"\n🎯 动作分布:")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        percentage = (count / len(all_actions)) * 100
        print(f"   - {name}: {count} 次 ({percentage:.1f}%)")
    
    return scores, action_counts

def visualize_results(scores, action_counts):
    """可视化结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 得分分布
    ax1.plot(scores, alpha=0.7)
    ax1.set_title('模型性能得分')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('得分')
    ax1.grid(True, alpha=0.3)
    
    # 得分直方图
    ax2.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('得分分布')
    ax2.set_xlabel('得分')
    ax2.set_ylabel('频次')
    ax2.grid(True, alpha=0.3)
    
    # 动作分布饼图
    action_names = ['接受', '拒绝', '阻止源', '阻止组']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    ax3.pie(action_counts, labels=action_names, autopct='%1.1f%%', colors=colors)
    ax3.set_title('动作分布')
    
    # 移动平均得分
    window_size = 10
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax4.plot(range(window_size-1, len(scores)), moving_avg, 'r-', linewidth=2, label=f'{window_size}回合移动平均')
        ax4.plot(scores, alpha=0.3, label='原始得分')
        ax4.set_title('移动平均得分')
        ax4.set_xlabel('回合')
        ax4.set_ylabel('得分')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def interactive_inference(agent):
    """交互式推理演示"""
    print(f"\n🎮 交互式推理演示")
    print(f"输入7个特征值（0-1之间），模型将预测最佳动作")
    print(f"输入 'quit' 退出")
    
    env = HTTPServerEnv(buffer_size=100, load_threshold=0.8, hazard_index=1)
    
    while True:
        try:
            user_input = input("\n请输入7个特征值（用空格分隔，0-1之间）: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            # 解析输入
            features = [float(x) for x in user_input.split()]
            
            if len(features) != 7:
                print("❌ 请输入7个特征值")
                continue
            
            # 验证特征值范围
            if not all(0 <= f <= 1 for f in features):
                print("❌ 特征值必须在0-1之间")
                continue
            
            # 转换为numpy数组
            state = np.array(features, dtype=np.float32)
            
            # 获取模型预测
            action = agent.choose_action(state)
            action_names = ['接受', '拒绝', '阻止源', '阻止组']
            
            print(f"📊 输入状态: {state}")
            print(f"🎯 模型预测动作: {action_names[action]} ({action})")
            
            # 获取Q值
            agent.q_func.eval()
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.q_func.device)
                q_values = agent.q_func(state_tensor).cpu().numpy()[0]
            
            print(f"💡 Q值: {q_values}")
            print(f"🔍 置信度: {np.max(q_values):.4f}")
            
        except ValueError:
            print("❌ 输入格式错误，请输入数字")
        except KeyboardInterrupt:
            break
    
    print("👋 退出交互式推理")

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 DDQN模型加载和使用示例")
    print("=" * 60)
    
    # 查找最新的模型文件
    model_dir = "models"
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        print("请先运行训练脚本生成模型")
        return
    
    # 查找.pth模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print(f"❌ 在 {model_dir} 目录中未找到模型文件")
        print("请先运行训练脚本生成模型")
        return
    
    # 选择最新的模型文件
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    print(f"📁 找到模型文件: {latest_model}")
    
    # 加载模型
    agent = load_trained_model(model_path)
    if agent is None:
        return
    
    # 测试模型
    scores, actions_taken = test_loaded_model(agent, n_episodes=50)
    
    # 分析性能
    scores, action_counts = analyze_model_performance(scores, actions_taken)
    
    # 可视化结果
    visualize_results(scores, action_counts)
    
    # 交互式推理
    interactive_inference(agent)
    
    print("\n🎉 模型使用示例完成！")

if __name__ == "__main__":
    main()
