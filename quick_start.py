#!/usr/bin/env python3
"""
RL-Cybersecurity 项目快速启动脚本
此脚本帮助您在模拟器和仿真器模式之间选择
"""

import os
import sys
import subprocess
import time
import signal
import threading

def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("🚀 RL-Cybersecurity: 基于强化学习的DDoS攻击检测系统")
    print("=" * 60)
    print()

def check_dependencies():
    """检查是否安装了所需的包"""
    try:
        import torch
        import gymnasium
        import numpy
        import matplotlib
        import psutil
        print("✅ 所有必需的包都已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def run_simulator():
    """运行仿真器模式"""
    print("🎮 启动仿真器模式...")
    print("此模式使用合成流量数据 - 非常适合测试！")
    print()
    
    try:
        subprocess.run([sys.executable, "example_simulator.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  仿真器被用户停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行仿真器时出错: {e}")

def run_emulator():
    """运行模拟器模式，启动多个进程"""
    print("🌐 启动模拟器模式...")
    print("此模式需要3个终端会话来处理真实网络流量")
    print()
    
    processes = []
    
    try:
        # 启动HTTP服务器
        print("1️⃣  启动HTTP服务器...")
        server_process = subprocess.Popen([sys.executable, "emulator/server.py"])
        processes.append(("HTTP服务器", server_process))
        time.sleep(2)
        
        # 启动训练智能体
        print("2️⃣  启动训练智能体...")
        agent_process = subprocess.Popen([sys.executable, "example_emulator.py"])
        processes.append(("训练智能体", agent_process))
        time.sleep(3)
        
        # 启动流量生成器
        print("3️⃣  启动流量生成器...")
        generator_process = subprocess.Popen([sys.executable, "start_generator.py"])
        processes.append(("流量生成器", generator_process))
        
        print("\n🎯 所有组件都在运行！")
        print("按 Ctrl+C 停止所有进程")
        print()
        
        # 等待用户停止
        while True:
            time.sleep(1)
            # 检查是否有进程意外死亡
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {name} 意外停止")
            
    except KeyboardInterrupt:
        print("\n⏹️  正在停止所有进程...")
        
        # 停止所有进程
        for name, proc in processes:
            print(f"🛑 正在停止 {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        print("✅ 所有进程已停止")

def show_help():
    """显示帮助信息"""
    print("用法:")
    print("  python quick_start.py [选项]")
    print()
    print("选项:")
    print("  simulator    在仿真器模式下运行（合成数据）")
    print("  emulator     在模拟器模式下运行（真实网络流量）")
    print("  help         显示此帮助消息")
    print()
    print("示例:")
    print("  python quick_start.py simulator")
    print("  python quick_start.py emulator")
    print()

def main():
    """主函数"""
    print_banner()
    
    if not check_dependencies():
        sys.exit(1)
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
        
        if option == "simulator":
            run_simulator()
        elif option == "emulator":
            run_emulator()
        elif option in ["help", "-h", "--help"]:
            show_help()
        else:
            print(f"❌ 未知选项: {option}")
            show_help()
            sys.exit(1)
    else:
        # 交互模式
        print("选择您的运行模式:")
        print("1. 🎮 仿真器模式（推荐给初学者）")
        print("2. 🌐 模拟器模式（真实网络流量）")
        print("3. ❓ 帮助")
        print("4. 🚪 退出")
        print()
        
        while True:
            try:
                choice = input("请输入您的选择 (1-4): ").strip()
                
                if choice == "1":
                    run_simulator()
                    break
                elif choice == "2":
                    run_emulator()
                    break
                elif choice == "3":
                    show_help()
                    print("\n选择您的运行模式:")
                    print("1. 🎮 仿真器模式（推荐给初学者）")
                    print("2. 🌐 模拟器模式（真实网络流量）")
                    print("3. ❓ 帮助")
                    print("4. 🚪 退出")
                    print()
                elif choice == "4":
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选择。请输入 1、2、3 或 4。")
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break

if __name__ == "__main__":
    main()
