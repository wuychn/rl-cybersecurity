#!/usr/bin/env python3
"""
RL-Cybersecurity 项目安装检查脚本
此脚本验证所有必需的包是否正确安装
"""

import sys
import importlib
import subprocess

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 正常")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - 需要Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', '未知')
        print(f"✅ {package_name} {version} - 正常")
        return True
    except ImportError:
        print(f"❌ {package_name} - 未安装")
        return False

def check_cuda():
    """检查CUDA可用性"""
    print("🚀 检查CUDA可用性...")
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA {cuda_version} - {device_count} 个设备可用")
            return True
        else:
            print("⚠️  CUDA不可用 - 将使用CPU（训练较慢）")
            return False
    except ImportError:
        print("❌ PyTorch未安装 - 无法检查CUDA")
        return False

def check_ports():
    """检查所需端口是否可用"""
    print("🔌 检查端口可用性...")
    import socket
    
    ports_to_check = [8070, 8080, 8090]
    available_ports = []
    
    for port in ports_to_check:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                available_ports.append(port)
                print(f"✅ 端口 {port} - 可用")
        except OSError:
            print(f"❌ 端口 {port} - 已被占用")
    
    return len(available_ports) == len(ports_to_check)

def run_quick_test():
    """运行快速测试以验证基本功能"""
    print("\n🧪 运行基本功能测试...")
    
    try:
        # 测试基本导入
        import numpy as np
        import gymnasium as gym
        from gymnasium import spaces
        
        # 测试环境创建
        obs_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        action_space = spaces.Discrete(4)
        
        print("✅ 基本环境设置 - 正常")
        
        # 测试神经网络创建
        import torch
        import torch.nn as nn
        
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(7, 4)
            
            def forward(self, x):
                return self.fc(x)
        
        net = TestNet()
        test_input = torch.randn(1, 7)
        output = net(test_input)
        
        print("✅ 神经网络创建 - 正常")
        print("✅ 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔍 RL-Cybersecurity 安装检查")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # 检查Python版本
    if not check_python_version():
        all_checks_passed = False
    
    print()
    
    # 检查必需的包
    print("📦 检查必需的包...")
    required_packages = [
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("psutil", "psutil"),
        ("prometheus-client", "prometheus_client"),
    ]
    
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_checks_passed = False
    
    print()
    
    # 检查CUDA
    if not check_cuda():
        print("注意: 没有CUDA训练会更慢")
    
    print()
    
    # 检查端口
    if not check_ports():
        print("警告: 某些端口已被占用")
        print("您可能需要停止其他服务或更改端口号")
    
    print()
    
    # 运行功能测试
    if not run_quick_test():
        all_checks_passed = False
    
    print()
    print("=" * 60)
    
    if all_checks_passed:
        print("🎉 安装检查成功完成！")
        print("✅ 您的系统已准备好运行RL-Cybersecurity")
        print()
        print("下一步:")
        print("1. 运行: python quick_start.py")
        print("2. 选择仿真器模式进行测试")
        print("3. 选择模拟器模式处理真实流量")
    else:
        print("❌ 安装检查发现问题")
        print()
        print("请修复上述问题并再次运行此脚本")
        print("常见解决方案:")
        print("- 运行: pip install -r requirements.txt")
        print("- 检查Python版本（需要3.8+）")
        print("- 释放端口8070、8080、8090")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
