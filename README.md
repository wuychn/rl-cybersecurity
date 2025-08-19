# RL-Cybersecurity: 基于强化学习的DDoS攻击检测系统

本项目实现了一个使用双深度Q网络（DDQN）的强化学习智能体，用于检测和缓解DDoS攻击。系统可以在两种模式下运行：**模拟器模式**（真实网络流量）和**仿真器模式**（合成流量模拟）。

## 项目结构

```
rl-cybersecurity/
├── agent/                    # 强化学习智能体实现
│   └── ddqn_agent.py        # 双深度Q网络智能体与神经网络
├── emulator/                 # 真实网络流量模拟
│   ├── enviroment.py        # 真实流量的Gym环境
│   ├── server.py            # 带系统监控的HTTP服务器
│   ├── message_generator.py # 流量生成器（用户vs机器人）
│   ├── features.py          # 网络数据特征提取
│   └── action_type.py       # 动作定义
├── simulator/                # 合成流量仿真
│   └── sim_env.py           # 仿真流量的Gym环境
├── example_emulator.py       # 模拟器模式训练脚本
├── example_simulator.py      # 仿真器模式训练脚本
├── start_generator.py        # 流量生成器启动器
├── training_log_emulator.txt # 模拟器训练日志示例
└── requirements.txt          # Python依赖包
```

## 功能特性

- **DDQN智能体**: 双深度Q网络用于最优动作选择
- **实时监控**: 系统资源使用率跟踪（CPU、内存）
- **流量分类**: 区分合法用户和攻击机器人
- **动作空间**: 4种动作（接受、拒绝、阻止源、阻止组）
- **状态空间**: 7维特征向量，包含流量模式和系统指标
- **奖励系统**: 惩罚误报/漏报，奖励正确分类

## 系统要求

- Python 3.8+
- CUDA兼容GPU（可选，用于加速训练）
- 模拟器模式需要网络访问

## 安装步骤

1. **克隆仓库**:
   ```bash
   git clone <仓库地址>
   cd rl-cybersecurity
   ```

2. **创建虚拟环境**（推荐）:
   ```bash
   python -m venv venv
   
   # Windows系统:
   venv\Scripts\activate
   
   # Linux/Mac系统:
   source venv/bin/activate
   ```

3. **安装依赖包**:
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**（可选但推荐）:
   ```bash
   python check_installation.py
   ```

## 运行项目

### 快速启动（推荐）

使用交互式快速启动脚本获得最佳体验：

```bash
python quick_start.py
```

这将引导您选择模拟器或仿真器模式。

### 选项1: 仿真器模式（推荐用于测试）

仿真器模式使用合成流量数据，非常适合初始测试和开发。

#### 方法1: 快速启动脚本
```bash
python quick_start.py simulator
```

#### 方法2: 直接执行
```bash
python example_simulator.py
```

#### 方法3: Windows批处理文件（Windows用户）
```cmd
run_simulator.bat
```

这将：
- 训练DDQN智能体1200个回合
- 测试训练好的智能体200个回合
- 显示训练和测试性能图表

### 选项2: 模拟器模式（真实网络流量）

模拟器模式处理真实网络流量，需要多个终端会话。

#### 方法1: 快速启动脚本（一键启动）
```bash
python quick_start.py emulator
```

#### 方法2: 手动设置（3个终端会话）

##### 终端1: 启动HTTP服务器
```bash
python emulator/server.py
```
- 服务器运行在 `127.0.0.1:8080`
- 监控系统资源（CPU、内存）
- 向智能体提供系统指标

##### 终端2: 启动训练智能体
```bash
python example_emulator.py
```
- 在 `127.0.0.1:8090` 创建代理服务器
- 连接到 `127.0.0.1:8080` 的HTTP服务器
- 训练DDQN智能体2000个回合
- 生成带时间戳的训练日志

##### 终端3: 启动流量生成器
```bash
python start_generator.py
```
- 在 `127.0.0.1:8070` 生成流量
- 向 `127.0.0.1:8090` 的智能体发送消息
- 35%合法用户流量，65%机器人流量
- 可配置消息间隔

#### 方法3: Windows批处理文件（Windows用户）
1. 在终端1中运行 `run_emulator.bat`（HTTP服务器）
2. 在终端2中运行 `python example_emulator.py`（训练智能体）
3. 在终端3中运行 `python start_generator.py`（流量生成器）

### 配置选项

#### 模拟器模式参数
- **代理主机/端口**: `127.0.0.1:8090`（默认）
- **服务器主机/端口**: `127.0.0.1:8080`（默认）
- **负载阈值**: 0.75（奖励计算的系统负载阈值）
- **危险指数**: 1.0（攻击检测的风险乘数）

#### 仿真器模式参数
- **缓冲区大小**: 100（请求缓冲区容量）
- **负载阈值**: 0.8（系统负载阈值）
- **用户请求概率**: 0.2（合法流量比例）
- **最大消息数**: 700（回合长度）

#### 训练参数
- **回合数**: 1200（仿真器）/ 2000（模拟器）
- **学习率**: 0.001
- **伽马**: 0.99（折扣因子）
- **探索率**: 1.0 → 0.01（探索衰减）
- **内存大小**: 200,000个转换
- **批次大小**: 128

## 理解输出结果

### 训练进度
训练脚本显示：
- 回合数和完成时间
- 预期总训练时间
- 当前回合得分和平均得分
- 训练日志保存到带时间戳的文件

### 性能指标
- **奖励**: 越高越好（正确分类）
- **误报**: 阻止合法用户（惩罚）
- **漏报**: 接受攻击（惩罚）
- **真阳性**: 正确阻止攻击（奖励）

### 可视化
- 训练性能随回合的变化
- 测试性能比较
- 奖励分布分析

## 故障排除

### 常见问题

1. **端口已被占用**:
   - 检查其他服务是否使用端口8070、8080或8090
   - 在相应文件中修改端口号

2. **连接被拒绝**:
   - 确保所有组件按正确顺序运行
   - 检查防火墙设置

3. **CUDA内存不足**:
   - 在 `ddqn_agent.py` 中减少批次大小
   - 通过修改设备选择使用仅CPU模式

4. **训练不收敛**:
   - 调整学习率或探索率衰减
   - 增加内存缓冲区大小
   - 修改奖励函数参数

5. **设备不匹配错误** (CUDA/CPU):
   - 运行 `python test_device_fix.py` 验证修复
   - 确保所有张量都在同一设备上
   - 检查PyTorch和CUDA版本兼容性

### 性能优化

- **GPU训练**: 确保正确安装CUDA以加速训练
- **内存管理**: 根据可用RAM调整缓冲区大小
- **网络延迟**: 开发时使用localhost，生产时使用真实IP

## 模型导出和使用

### 自动模型保存
训练完成后，系统会自动保存模型到 `models/` 目录：
- **PyTorch模型** (`.pth`): 包含网络权重、优化器状态和训练参数
- **ONNX模型** (`.onnx`): 用于生产部署的标准化格式
- **模型信息** (`.json`): 包含模型元数据和训练统计

### 模型文件结构
```
models/
├── ddqn_model_YYYYMMDD_HHMMSS.pth      # PyTorch模型
├── ddqn_model_YYYYMMDD_HHMMSS.onnx     # ONNX模型
└── ddqn_model_YYYYMMDD_HHMMSS_info.json # 模型信息
```

### 加载和使用模型

#### 方法1: 使用加载脚本
```bash
python load_and_use_model.py
```
- 自动查找最新模型
- 测试模型性能
- 交互式推理演示
- 性能可视化分析

#### 方法2: 生产环境部署
```bash
python deploy_model.py
```
- 模型基准测试
- PyTorch和ONNX性能对比
- 生产服务模拟
- 实时推理统计

#### 方法3: 编程方式使用
```python
from agent.ddqn_agent import DoubleQAgent

# 创建智能体
agent = DoubleQAgent(observation_space_shape=7, action_space_n=4)

# 加载训练好的模型
agent.load_model("models/ddqn_model_latest.pth")

# 进行预测
state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
action = agent.choose_action(state)
```

### 模型部署选项

#### PyTorch部署
- **优点**: 完整功能、易于调试、支持GPU加速
- **适用**: 开发环境、研究用途、需要模型修改

#### ONNX部署
- **优点**: 跨平台、高性能、标准化、易于集成
- **适用**: 生产环境、边缘设备、多语言调用

### 模型性能优化
- **批处理推理**: 支持批量输入提高吞吐量
- **模型量化**: 减少内存占用和推理时间
- **GPU加速**: 自动检测和使用CUDA设备
- **缓存优化**: 优化器状态和中间结果缓存

### 模型管理工具
```bash
# 列出所有模型
python model_manager.py

# 显示模型详细信息
python model_manager.py --info ddqn_model_20241201

# 比较所有模型
python model_manager.py --compare

# 导出模型摘要
python model_manager.py --export models_summary.json

# 清理旧模型（保留最新的3个）
python model_manager.py --cleanup 3
```

## 自定义配置

### 添加新动作
1. 修改 `emulator/action_type.py`
2. 在环境文件中更新奖励函数
3. 调整神经网络输出大小

### 修改特征
1. 编辑 `emulator/features.py`
2. 更新观察空间维度
3. 重新训练智能体

### 更改奖励函数
1. 修改环境类中的 `get_reward()` 方法
2. 调整惩罚/奖励权重
3. 在不同场景下测试

## 贡献指南

1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 如果适用，添加测试
5. 提交拉取请求

## 许可证

[在此添加您的许可证信息]

## 快速参考

### 常用命令
```bash
# 快速启动（交互式）
python quick_start.py

# 仿真器模式
python quick_start.py simulator
python example_simulator.py

# 模拟器模式（一键启动）
python quick_start.py emulator

# 手动模拟器设置
python emulator/server.py          # 终端1
python example_emulator.py         # 终端2  
python start_generator.py          # 终端3

# 测试和验证
python check_installation.py       # 检查安装

# 模型管理
python load_and_use_model.py       # 加载和使用模型
python deploy_model.py             # 生产环境部署
python model_manager.py            # 模型管理工具
```

### 端口配置
- **HTTP服务器**: `127.0.0.1:8080`
- **训练智能体**: `127.0.0.1:8090`
- **流量生成器**: `127.0.0.1:8070`

### 文件位置
- **训练日志**: `training_log_YYYYMMDD_HHMMSS.txt`
- **智能体模型**: 训练期间存储在内存中
- **配置**: 修改环境文件中的参数

## 致谢

- Gymnasium提供的强化学习环境框架
- PyTorch提供的深度学习能力
- 网络安全研究社区的启发
