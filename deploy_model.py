#!/usr/bin/env python3  # 指定Python解释器路径，确保脚本在Unix/Linux系统上可执行
"""
生产环境模型部署脚本
将训练好的DDQN模型部署为可用的推理服务
"""

import os  # 导入操作系统接口模块，用于文件路径操作
import sys  # 导入系统相关参数和函数模块
import time  # 导入时间处理模块，用于性能计时和统计
import json  # 导入JSON数据处理模块，用于读取模型信息文件
import numpy as np  # 导入数值计算库，用于数组操作和数学计算
from typing import Dict, List, Tuple, Optional  # 导入类型提示模块，用于函数参数和返回值类型声明

# 添加项目路径到Python路径中，确保可以导入项目内的模块
sys.path.append('.')

try:  # 尝试导入必需的深度学习库
    import torch  # 导入PyTorch深度学习框架
    import onnxruntime as ort  # 导入ONNX运行时，用于ONNX模型推理
except ImportError as e:  # 如果导入失败，捕获异常
    print(f"❌ 缺少必需的包: {e}")  # 打印错误信息
    print("请安装: pip install torch onnxruntime")  # 提示用户安装缺失的包
    sys.exit(1)  # 退出程序，返回错误码1

class ModelDeployer:  # 定义模型部署器主类
    """模型部署器 - 负责管理不同格式模型的部署"""
    
    def __init__(self, model_path: str):  # 初始化方法，接收模型文件路径参数
        """初始化部署器
        
        Args:
            model_path: PyTorch模型文件路径
        """
        self.model_path = model_path  # 存储模型文件路径
        self.model_dir = os.path.dirname(model_path)  # 获取模型文件所在目录
        self.model_name = os.path.basename(model_path)  # 获取模型文件名
        
        # 检查是否存在对应的ONNX模型文件
        onnx_path = model_path.replace('.pth', '.onnx')  # 将.pth扩展名替换为.onnx
        if os.path.exists(onnx_path):  # 如果ONNX文件存在
            self.onnx_path = onnx_path  # 设置ONNX文件路径
        else:  # 如果ONNX文件不存在
            self.onnx_path = None  # 将ONNX路径设为None
        
        # 加载模型信息（从JSON文件）
        self.model_info = self._load_model_info()  # 调用私有方法加载模型信息
        
        print(f"🚀 模型部署器初始化完成")  # 打印初始化完成信息
        print(f"📁 模型路径: {model_path}")  # 打印模型路径
        print(f"🔧 模型类型: {self.model_info.get('model_type', 'Unknown')}")  # 打印模型类型
    
    def _load_model_info(self) -> Dict:  # 私有方法，加载模型信息，返回字典类型
        """加载模型信息 - 从JSON文件中读取模型的元数据信息"""
        info_path = self.model_path.replace('.pth', '_info.json')  # 构造信息文件路径
        if os.path.exists(info_path):  # 如果信息文件存在
            with open(info_path, 'r', encoding='utf-8') as f:  # 以UTF-8编码打开文件
                return json.load(f)  # 解析JSON内容并返回
        return {}  # 如果文件不存在，返回空字典
    
    def deploy_pytorch(self) -> 'PyTorchDeployer':  # 部署PyTorch模型，返回PyTorch部署器实例
        """部署PyTorch模型 - 创建并返回PyTorch模型部署器"""
        return PyTorchDeployer(self.model_path, self.model_info)  # 创建PyTorch部署器实例
    
    def deploy_onnx(self) -> Optional['ONNXDeployer']:  # 部署ONNX模型，可能返回None
        """部署ONNX模型 - 如果ONNX模型存在则创建ONNX部署器"""
        if self.onnx_path and os.path.exists(self.onnx_path):  # 如果ONNX路径存在且文件存在
            return ONNXDeployer(self.onnx_path, self.model_info)  # 创建ONNX部署器实例
        else:  # 如果ONNX模型不存在
            print("⚠️  ONNX模型不存在，无法部署ONNX版本")  # 打印警告信息
            return None  # 返回None
    
    def benchmark_models(self, test_data: np.ndarray, n_runs: int = 1000) -> Dict:  # 基准测试方法
        """基准测试不同部署方式 - 比较PyTorch和ONNX模型的推理性能"""
        print(f"\n📊 开始模型基准测试 ({n_runs} 次推理)...")  # 打印测试开始信息
        
        results = {}  # 初始化结果字典
        
        # PyTorch基准测试
        pytorch_deployer = self.deploy_pytorch()  # 创建PyTorch部署器
        pytorch_times = []  # 初始化PyTorch推理时间列表
        
        for _ in range(n_runs):  # 循环执行指定次数的推理
            start_time = time.perf_counter()  # 记录开始时间（高精度计时器）
            _ = pytorch_deployer.predict(test_data)  # 执行推理（忽略返回值）
            end_time = time.perf_counter()  # 记录结束时间
            pytorch_times.append((end_time - start_time) * 1000)  # 计算推理时间并转换为毫秒
        
        results['pytorch'] = {  # 存储PyTorch测试结果
            'mean_time': np.mean(pytorch_times),  # 平均推理时间
            'std_time': np.std(pytorch_times),  # 推理时间标准差
            'min_time': np.min(pytorch_times),  # 最小推理时间
            'max_time': np.max(pytorch_times)  # 最大推理时间
        }
        
        # ONNX基准测试
        onnx_deployer = self.deploy_onnx()  # 尝试创建ONNX部署器
        if onnx_deployer:  # 如果ONNX部署器创建成功
            onnx_times = []  # 初始化ONNX推理时间列表
            
            for _ in range(n_runs):  # 循环执行指定次数的推理
                start_time = time.perf_counter()  # 记录开始时间
                _ = onnx_deployer.predict(test_data)  # 执行推理
                end_time = time.perf_counter()  # 记录结束时间
                onnx_times.append((end_time - start_time) * 1000)  # 计算推理时间并转换为毫秒
            
            results['onnx'] = {  # 存储ONNX测试结果
                'mean_time': np.mean(onnx_times),  # 平均推理时间
                'std_time': np.std(onnx_times),  # 推理时间标准差
                'min_time': np.min(onnx_times),  # 最小推理时间
                'max_time': np.max(onnx_times)  # 最大推理时间
            }
        
        # 打印结果
        print(f"\n📈 基准测试结果:")  # 打印结果标题
        print(f"PyTorch:")  # 打印PyTorch结果
        print(f"  平均推理时间: {results['pytorch']['mean_time']:.2f} ms")  # 打印平均时间
        print(f"  标准差: {results['pytorch']['std_time']:.2f} ms")  # 打印标准差
        
        if 'onnx' in results:  # 如果ONNX结果存在
            print(f"ONNX:")  # 打印ONNX结果
            print(f"  平均推理时间: {results['onnx']['mean_time']:.2f} ms")  # 打印平均时间
            print(f"  标准差: {results['onnx']['std_time']:.2f} ms")  # 打印标准差
            
            speedup = results['pytorch']['mean_time'] / results['onnx']['mean_time']  # 计算加速比
            print(f"🚀 ONNX加速比: {speedup:.2f}x")  # 打印加速比
        
        return results  # 返回测试结果

class PyTorchDeployer:  # 定义PyTorch模型部署器类
    """PyTorch模型部署器 - 专门处理PyTorch格式的模型"""
    
    def __init__(self, model_path: str, model_info: Dict):  # 初始化方法
        """初始化PyTorch部署器 - 加载PyTorch模型到指定设备"""
        self.model_path = model_path  # 存储模型文件路径
        self.model_info = model_info  # 存储模型信息
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU设备
        self.model = self._load_model()  # 调用私有方法加载模型
        
        print(f"✅ PyTorch模型加载完成 (设备: {self.device})")  # 打印加载完成信息
    
    def _load_model(self):  # 私有方法，加载PyTorch模型
        """加载PyTorch模型 - 从检查点文件恢复模型状态"""
        from agent.ddqn_agent import QNN  # 导入Q网络模型类
        
        checkpoint = torch.load(self.model_path, map_location=self.device)  # 加载检查点文件到指定设备
        
        # 创建模型
        model = QNN(  # 实例化Q网络模型
            checkpoint['observation_space_shape'],  # 观察空间形状
            checkpoint['action_space_n'],  # 动作空间大小
            seed=42  # 随机种子
        ).to(self.device)  # 将模型移动到指定设备
        
        # 加载权重
        model.load_state_dict(checkpoint['q_func_state_dict'])  # 加载模型权重
        model.eval()  # 设置为评估模式（关闭梯度计算和dropout）
        
        return model  # 返回加载好的模型
    
    def predict(self, input_data: np.ndarray) -> Tuple[int, np.ndarray]:  # 预测方法
        """进行预测 - 使用PyTorch模型进行推理
        
        Args:
            input_data: 输入特征 (7维)
            
        Returns:
            (预测动作, Q值数组)
        """
        with torch.no_grad():  # 关闭梯度计算，提高推理效率
            # 转换为张量
            if isinstance(input_data, np.ndarray):  # 如果输入是numpy数组
                input_tensor = torch.from_numpy(input_data).float().to(self.device)  # 转换为PyTorch张量并移动到设备
            else:  # 如果输入已经是张量
                input_tensor = input_data.to(self.device)  # 直接移动到设备
            
            # 确保正确的形状
            if input_tensor.dim() == 1:  # 如果是一维张量
                input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
            
            # 前向传播
            q_values = self.model(input_tensor)  # 模型前向传播，得到Q值
            
            # 选择最佳动作
            action = torch.argmax(q_values, dim=1).item()  # 选择Q值最大的动作索引
            
            return action, q_values.cpu().numpy()[0]  # 返回动作和Q值数组

class ONNXDeployer:  # 定义ONNX模型部署器类
    """ONNX模型部署器 - 专门处理ONNX格式的模型"""
    
    def __init__(self, onnx_path: str, model_info: Dict):  # 初始化方法
        """初始化ONNX部署器 - 创建ONNX推理会话"""
        self.onnx_path = onnx_path  # 存储ONNX模型文件路径
        self.model_info = model_info  # 存储模型信息
        
        # 创建推理会话
        self.session = ort.InferenceSession(onnx_path)  # 创建ONNX推理会话
        self.input_name = self.session.get_inputs()[0].name  # 获取输入节点名称
        self.output_name = self.session.get_outputs()[0].name  # 获取输出节点名称
        
        print(f"✅ ONNX模型加载完成")  # 打印加载完成信息
        print(f"  输入名称: {self.input_name}")  # 打印输入节点名称
        print(f"  输出名称: {self.output_name}")  # 打印输出节点名称
    
    def predict(self, input_data: np.ndarray) -> Tuple[int, np.ndarray]:  # 预测方法
        """进行预测 - 使用ONNX模型进行推理
        
        Args:
            input_data: 输入特征 (7维)
            
        Returns:
            (预测动作, Q值数组)
        """
        # 确保正确的形状和类型
        if input_data.dim() == 1:  # 如果是一维数组
            input_data = input_data.reshape(1, -1)  # 重塑为二维数组（添加批次维度）
        
        input_data = input_data.astype(np.float32)  # 转换为float32类型（ONNX要求）
        
        # 运行推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})  # 执行ONNX推理
        q_values = outputs[0][0]  # 获取输出结果（第一个输出的第一个批次）
        
        # 选择最佳动作
        action = np.argmax(q_values)  # 选择Q值最大的动作索引
        
        return action, q_values  # 返回动作和Q值数组

class ProductionService:  # 定义生产环境服务类
    """生产环境服务 - 提供模型推理的Web服务接口"""
    
    def __init__(self, deployer: PyTorchDeployer):  # 初始化方法
        """初始化生产服务 - 设置服务统计信息"""
        self.deployer = deployer  # 存储模型部署器
        self.request_count = 0  # 初始化请求计数器
        self.start_time = time.time()  # 记录服务启动时间
        
        print(f"🏭 生产环境服务启动")  # 打印服务启动信息
    
    def process_request(self, features: List[float]) -> Dict:  # 处理请求方法
        """处理推理请求 - 验证输入并返回预测结果
        
        Args:
            features: 7维特征向量
            
        Returns:
            包含预测结果的字典
        """
        start_time = time.perf_counter()  # 记录请求开始时间
        
        # 验证输入
        if len(features) != 7:  # 检查特征维度是否为7
            return {  # 返回错误信息
                'error': '特征维度必须为7',  # 错误描述
                'received': len(features)  # 实际接收到的维度
            }
        
        if not all(0 <= f <= 1 for f in features):  # 检查所有特征值是否在0-1范围内
            return {  # 返回错误信息
                'error': '特征值必须在0-1之间',  # 错误描述
                'received': features  # 实际接收到的特征值
            }
        
        # 转换为numpy数组
        input_data = np.array(features, dtype=np.float32)  # 将特征列表转换为numpy数组
        
        # 进行预测
        action, q_values = self.deployer.predict(input_data)  # 调用部署器进行预测
        
        # 计算推理时间
        inference_time = (time.perf_counter() - start_time) * 1000  # 计算推理时间并转换为毫秒
        
        # 更新统计信息
        self.request_count += 1  # 增加请求计数
        
        # 返回结果
        action_names = ['接受', '拒绝', '阻止源', '阻止组']  # 动作名称映射
        
        return {  # 返回成功结果
            'success': True,  # 成功标志
            'action': action,  # 预测的动作索引
            'action_name': action_names[action],  # 动作名称
            'q_values': q_values.tolist(),  # Q值数组（转换为列表）
            'confidence': float(np.max(q_values)),  # 置信度（最大Q值）
            'inference_time_ms': inference_time,  # 推理时间（毫秒）
            'request_id': self.request_count,  # 请求ID
            'timestamp': time.time()  # 时间戳
        }
    
    def get_stats(self) -> Dict:  # 获取统计信息方法
        """获取服务统计信息 - 返回服务的运行统计"""
        uptime = time.time() - self.start_time  # 计算服务运行时间
        
        return {  # 返回统计信息
            'uptime_seconds': uptime,  # 运行时间（秒）
            'total_requests': self.request_count,  # 总请求数
            'requests_per_second': self.request_count / uptime if uptime > 0 else 0,  # 每秒请求数
            'start_time': self.start_time  # 启动时间
        }

def main():  # 主函数
    """主函数 - 程序入口点，执行模型部署和测试"""
    print("=" * 60)  # 打印分隔线
    print("🚀 DDQN模型生产环境部署")  # 打印程序标题
    print("=" * 60)  # 打印分隔线
    
    # 查找最新的模型文件
    model_dir = "models"  # 模型目录名称
    if not os.path.exists(model_dir):  # 如果模型目录不存在
        print(f"❌ 模型目录不存在: {model_dir}")  # 打印错误信息
        print("请先运行训练脚本生成模型")  # 提示用户
        return  # 退出函数
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]  # 获取所有.pth模型文件
    if not model_files:  # 如果没有找到模型文件
        print(f"❌ 在 {model_dir} 目录中未找到模型文件")  # 打印错误信息
        print("请先运行训练脚本生成模型")  # 提示用户
        return  # 退出函数
    
    # 选择最新的模型文件
    latest_model = sorted(model_files)[-1]  # 按文件名排序，选择最后一个（最新的）
    model_path = os.path.join(model_dir, latest_model)  # 构造完整的模型文件路径
    print(f"📁 使用模型: {latest_model}")  # 打印使用的模型文件名
    
    # 创建部署器
    deployer = ModelDeployer(model_path)  # 创建模型部署器实例
    
    # 基准测试
    test_data = np.random.rand(7).astype(np.float32)  # 生成随机测试数据（7维）
    benchmark_results = deployer.benchmark_models(test_data, n_runs=1000)  # 执行基准测试
    
    # 部署PyTorch模型
    pytorch_deployer = deployer.deploy_pytorch()  # 创建PyTorch部署器
    
    # 创建生产服务
    service = ProductionService(pytorch_deployer)  # 创建生产环境服务实例
    
    # 交互式测试
    print(f"\n🎮 交互式生产服务测试")  # 打印测试说明
    print(f"输入7个特征值（0-1之间），服务将返回预测结果")  # 打印使用说明
    print(f"输入 'quit' 退出，输入 'stats' 查看统计信息")  # 打印命令说明
    
    while True:  # 无限循环，等待用户输入
        try:  # 尝试执行用户输入处理
            user_input = input("\n请输入7个特征值（用空格分隔）: ").strip()  # 获取用户输入并去除首尾空格
            
            if user_input.lower() == 'quit':  # 如果用户输入quit
                break  # 跳出循环
            
            if user_input.lower() == 'stats':  # 如果用户输入stats
                stats = service.get_stats()  # 获取服务统计信息
                print(f"\n📊 服务统计信息:")  # 打印统计信息标题
                print(f"  运行时间: {stats['uptime_seconds']:.1f} 秒")  # 打印运行时间
                print(f"  总请求数: {stats['total_requests']}")  # 打印总请求数
                print(f"  请求/秒: {stats['requests_per_second']:.2f}")  # 打印每秒请求数
                continue  # 继续下一次循环
            
            # 解析输入
            features = [float(x) for x in user_input.split()]  # 将输入字符串分割并转换为浮点数列表
            
            if len(features) != 7:  # 如果特征数量不是7个
                print("❌ 请输入7个特征值")  # 打印错误提示
                continue  # 继续下一次循环
            
            # 处理请求
            result = service.process_request(features)  # 调用服务处理请求
            
            if result.get('success'):  # 如果请求处理成功
                print(f"✅ 预测成功!")  # 打印成功信息
                print(f"🎯 动作: {result['action_name']} ({result['action']})")  # 打印预测动作
                print(f"💡 置信度: {result['confidence']:.4f}")  # 打印置信度
                print(f"⏱️  推理时间: {result['inference_time_ms']:.2f} ms")  # 打印推理时间
                print(f"🆔 请求ID: {result['request_id']}")  # 打印请求ID
            else:  # 如果请求处理失败
                print(f"❌ 预测失败: {result.get('error')}")  # 打印错误信息
                
        except ValueError:  # 捕获值错误异常（如输入非数字）
            print("❌ 输入格式错误，请输入数字")  # 打印错误提示
        except KeyboardInterrupt:  # 捕获键盘中断异常（Ctrl+C）
            break  # 跳出循环
    
    # 最终统计
    final_stats = service.get_stats()  # 获取最终统计信息
    print(f"\n📊 最终统计:")  # 打印最终统计标题
    print(f"  总请求数: {final_stats['total_requests']}")  # 打印总请求数
    print(f"  平均请求/秒: {final_stats['requests_per_second']:.2f}")  # 打印平均每秒请求数
    
    print("\n🎉 生产环境部署测试完成！")  # 打印完成信息

if __name__ == "__main__":  # 如果脚本直接运行（不是被导入）
    main()  # 调用主函数
