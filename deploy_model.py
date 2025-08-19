#!/usr/bin/env python3
"""
生产环境模型部署脚本
将训练好的DDQN模型部署为可用的推理服务
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append('.')

try:
    import torch
    import onnxruntime as ort
except ImportError as e:
    print(f"❌ 缺少必需的包: {e}")
    print("请安装: pip install torch onnxruntime")
    sys.exit(1)

class ModelDeployer:
    """模型部署器"""
    
    def __init__(self, model_path: str):
        """初始化部署器
        
        Args:
            model_path: PyTorch模型文件路径
        """
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.model_name = os.path.basename(model_path)
        
        # 检查ONNX模型
        onnx_path = model_path.replace('.pth', '.onnx')
        if os.path.exists(onnx_path):
            self.onnx_path = onnx_path
        else:
            self.onnx_path = None
        
        # 加载模型信息
        self.model_info = self._load_model_info()
        
        print(f"🚀 模型部署器初始化完成")
        print(f"📁 模型路径: {model_path}")
        print(f"🔧 模型类型: {self.model_info.get('model_type', 'Unknown')}")
    
    def _load_model_info(self) -> Dict:
        """加载模型信息"""
        info_path = self.model_path.replace('.pth', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def deploy_pytorch(self) -> 'PyTorchDeployer':
        """部署PyTorch模型"""
        return PyTorchDeployer(self.model_path, self.model_info)
    
    def deploy_onnx(self) -> Optional['ONNXDeployer']:
        """部署ONNX模型"""
        if self.onnx_path and os.path.exists(self.onnx_path):
            return ONNXDeployer(self.onnx_path, self.model_info)
        else:
            print("⚠️  ONNX模型不存在，无法部署ONNX版本")
            return None
    
    def benchmark_models(self, test_data: np.ndarray, n_runs: int = 1000) -> Dict:
        """基准测试不同部署方式"""
        print(f"\n📊 开始模型基准测试 ({n_runs} 次推理)...")
        
        results = {}
        
        # PyTorch基准测试
        pytorch_deployer = self.deploy_pytorch()
        pytorch_times = []
        
        for _ in range(n_runs):
            start_time = time.perf_counter()
            _ = pytorch_deployer.predict(test_data)
            end_time = time.perf_counter()
            pytorch_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        results['pytorch'] = {
            'mean_time': np.mean(pytorch_times),
            'std_time': np.std(pytorch_times),
            'min_time': np.min(pytorch_times),
            'max_time': np.max(pytorch_times)
        }
        
        # ONNX基准测试
        onnx_deployer = self.deploy_onnx()
        if onnx_deployer:
            onnx_times = []
            
            for _ in range(n_runs):
                start_time = time.perf_counter()
                _ = onnx_deployer.predict(test_data)
                end_time = time.perf_counter()
                onnx_times.append((end_time - start_time) * 1000)
            
            results['onnx'] = {
                'mean_time': np.mean(onnx_times),
                'std_time': np.std(onnx_times),
                'min_time': np.min(onnx_times),
                'max_time': np.max(onnx_times)
            }
        
        # 打印结果
        print(f"\n📈 基准测试结果:")
        print(f"PyTorch:")
        print(f"  平均推理时间: {results['pytorch']['mean_time']:.2f} ms")
        print(f"  标准差: {results['pytorch']['std_time']:.2f} ms")
        
        if 'onnx' in results:
            print(f"ONNX:")
            print(f"  平均推理时间: {results['onnx']['mean_time']:.2f} ms")
            print(f"  标准差: {results['onnx']['std_time']:.2f} ms")
            
            speedup = results['pytorch']['mean_time'] / results['onnx']['mean_time']
            print(f"🚀 ONNX加速比: {speedup:.2f}x")
        
        return results

class PyTorchDeployer:
    """PyTorch模型部署器"""
    
    def __init__(self, model_path: str, model_info: Dict):
        """初始化PyTorch部署器"""
        self.model_path = model_path
        self.model_info = model_info
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        print(f"✅ PyTorch模型加载完成 (设备: {self.device})")
    
    def _load_model(self):
        """加载PyTorch模型"""
        from agent.ddqn_agent import QNN
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 创建模型
        model = QNN(
            checkpoint['observation_space_shape'],
            checkpoint['action_space_n'],
            seed=42
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['q_func_state_dict'])
        model.eval()
        
        return model
    
    def predict(self, input_data: np.ndarray) -> Tuple[int, np.ndarray]:
        """进行预测
        
        Args:
            input_data: 输入特征 (7维)
            
        Returns:
            (预测动作, Q值数组)
        """
        with torch.no_grad():
            # 转换为张量
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float().to(self.device)
            else:
                input_tensor = input_data.to(self.device)
            
            # 确保正确的形状
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # 前向传播
            q_values = self.model(input_tensor)
            
            # 选择最佳动作
            action = torch.argmax(q_values, dim=1).item()
            
            return action, q_values.cpu().numpy()[0]

class ONNXDeployer:
    """ONNX模型部署器"""
    
    def __init__(self, onnx_path: str, model_info: Dict):
        """初始化ONNX部署器"""
        self.onnx_path = onnx_path
        self.model_info = model_info
        
        # 创建推理会话
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ ONNX模型加载完成")
        print(f"  输入名称: {self.input_name}")
        print(f"  输出名称: {self.output_name}")
    
    def predict(self, input_data: np.ndarray) -> Tuple[int, np.ndarray]:
        """进行预测
        
        Args:
            input_data: 输入特征 (7维)
            
        Returns:
            (预测动作, Q值数组)
        """
        # 确保正确的形状和类型
        if input_data.dim() == 1:
            input_data = input_data.reshape(1, -1)
        
        input_data = input_data.astype(np.float32)
        
        # 运行推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        q_values = outputs[0][0]
        
        # 选择最佳动作
        action = np.argmax(q_values)
        
        return action, q_values

class ProductionService:
    """生产环境服务"""
    
    def __init__(self, deployer: PyTorchDeployer):
        """初始化生产服务"""
        self.deployer = deployer
        self.request_count = 0
        self.start_time = time.time()
        
        print(f"🏭 生产环境服务启动")
    
    def process_request(self, features: List[float]) -> Dict:
        """处理请求
        
        Args:
            features: 7维特征向量
            
        Returns:
            包含预测结果的字典
        """
        start_time = time.perf_counter()
        
        # 验证输入
        if len(features) != 7:
            return {
                'error': '特征维度必须为7',
                'received': len(features)
            }
        
        if not all(0 <= f <= 1 for f in features):
            return {
                'error': '特征值必须在0-1之间',
                'received': features
            }
        
        # 转换为numpy数组
        input_data = np.array(features, dtype=np.float32)
        
        # 进行预测
        action, q_values = self.deployer.predict(input_data)
        
        # 计算推理时间
        inference_time = (time.perf_counter() - start_time) * 1000  # 毫秒
        
        # 更新统计信息
        self.request_count += 1
        
        # 返回结果
        action_names = ['接受', '拒绝', '阻止源', '阻止组']
        
        return {
            'success': True,
            'action': action,
            'action_name': action_names[action],
            'q_values': q_values.tolist(),
            'confidence': float(np.max(q_values)),
            'inference_time_ms': inference_time,
            'request_id': self.request_count,
            'timestamp': time.time()
        }
    
    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'requests_per_second': self.request_count / uptime if uptime > 0 else 0,
            'start_time': self.start_time
        }

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 DDQN模型生产环境部署")
    print("=" * 60)
    
    # 查找最新的模型文件
    model_dir = "models"
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        print("请先运行训练脚本生成模型")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print(f"❌ 在 {model_dir} 目录中未找到模型文件")
        print("请先运行训练脚本生成模型")
        return
    
    # 选择最新的模型文件
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    print(f"📁 使用模型: {latest_model}")
    
    # 创建部署器
    deployer = ModelDeployer(model_path)
    
    # 基准测试
    test_data = np.random.rand(7).astype(np.float32)
    benchmark_results = deployer.benchmark_models(test_data, n_runs=1000)
    
    # 部署PyTorch模型
    pytorch_deployer = deployer.deploy_pytorch()
    
    # 创建生产服务
    service = ProductionService(pytorch_deployer)
    
    # 交互式测试
    print(f"\n🎮 交互式生产服务测试")
    print(f"输入7个特征值（0-1之间），服务将返回预测结果")
    print(f"输入 'quit' 退出，输入 'stats' 查看统计信息")
    
    while True:
        try:
            user_input = input("\n请输入7个特征值（用空格分隔）: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'stats':
                stats = service.get_stats()
                print(f"\n📊 服务统计信息:")
                print(f"  运行时间: {stats['uptime_seconds']:.1f} 秒")
                print(f"  总请求数: {stats['total_requests']}")
                print(f"  请求/秒: {stats['requests_per_second']:.2f}")
                continue
            
            # 解析输入
            features = [float(x) for x in user_input.split()]
            
            if len(features) != 7:
                print("❌ 请输入7个特征值")
                continue
            
            # 处理请求
            result = service.process_request(features)
            
            if result.get('success'):
                print(f"✅ 预测成功!")
                print(f"🎯 动作: {result['action_name']} ({result['action']})")
                print(f"💡 置信度: {result['confidence']:.4f}")
                print(f"⏱️  推理时间: {result['inference_time_ms']:.2f} ms")
                print(f"🆔 请求ID: {result['request_id']}")
            else:
                print(f"❌ 预测失败: {result.get('error')}")
                
        except ValueError:
            print("❌ 输入格式错误，请输入数字")
        except KeyboardInterrupt:
            break
    
    # 最终统计
    final_stats = service.get_stats()
    print(f"\n📊 最终统计:")
    print(f"  总请求数: {final_stats['total_requests']}")
    print(f"  平均请求/秒: {final_stats['requests_per_second']:.2f}")
    
    print("\n🎉 生产环境部署测试完成！")

if __name__ == "__main__":
    main()
