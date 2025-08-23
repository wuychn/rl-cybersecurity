#!/usr/bin/env python3
"""
模型管理工具
用于管理、查看和比较训练好的DDQN模型
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

def list_models(model_dir: str = "models") -> List[Dict]:
    """列出所有可用的模型"""
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return []
    
    models = []
    
    # 查找所有.pth文件
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            model_path = os.path.join(model_dir, filename)
            model_info = load_model_info(model_path)
            
            # 获取文件信息
            stat = os.stat(model_path)
            file_size = stat.st_size / (1024 * 1024)  # MB
            
            models.append({
                'filename': filename,
                'path': model_path,
                'size_mb': file_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'info': model_info
            })
    
    # 按修改时间排序（最新的在前）
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    return models

def load_model_info(model_path: str) -> Dict:
    """加载模型信息文件"""
    info_path = model_path.replace('.pth', '_info.json')
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  无法加载模型信息: {e}")
    
    return {}

def display_model_info(model: Dict):
    """显示模型信息"""
    print(f"\n📁 模型文件: {model['filename']}")
    print(f"📊 文件大小: {model['size_mb']:.2f} MB")
    print(f"🕒 修改时间: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    if model['info']:
        info = model['info']
        print(f"🔧 模型类型: {info.get('model_type', 'Unknown')}")
        print(f"📅 保存时间: {info.get('saved_at', 'Unknown')}")
        
        if 'parameters' in info:
            params = info['parameters']
            print(f"📐 观察空间: {params.get('observation_space', 'Unknown')}")
            print(f"🎯 动作空间: {params.get('action_space', 'Unknown')}")
            print(f"🎲 折扣因子: {params.get('gamma', 'Unknown')}")
            print(f"🔍 探索率: {params.get('epsilon', 'Unknown')}")
            print(f"📚 学习率: {params.get('learning_rate', 'Unknown')}")
    else:
        print("⚠️  无模型信息文件")

def compare_models(models: List[Dict]):
    """比较多个模型"""
    if len(models) < 2:
        print("❌ 需要至少2个模型进行比较")
        return
    
    print(f"\n🔍 模型比较 ({len(models)} 个模型)")
    print("=" * 80)
    
    # 表头
    headers = ['文件名', '大小(MB)', '修改时间', '观察空间', '动作空间', '学习率']
    row_format = "{:<25} {:<10} {:<20} {:<12} {:<12} {:<10}"
    
    print(row_format.format(*headers))
    print("-" * 80)
    
    # 数据行
    for model in models:
        info = model['info']
        params = info.get('parameters', {})
        
        row = [
            model['filename'][:24],
            f"{model['size_mb']:.1f}",
            model['modified'].strftime('%Y-%m-%d %H:%M'),
            str(params.get('observation_space', 'N/A')),
            str(params.get('action_space', 'N/A')),
            str(params.get('learning_rate', 'N/A'))
        ]
        
        print(row_format.format(*row))

def export_model_summary(models: List[Dict], output_file: str):
    """导出模型摘要到文件"""
    summary = {
        'export_time': datetime.now().isoformat(),
        'total_models': len(models),
        'models': []
    }
    
    for model in models:
        model_summary = {
            'filename': model['filename'],
            'size_mb': model['size_mb'],
            'modified': model['modified'].isoformat(),
            'info': model['info']
        }
        summary['models'].append(model_summary)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 模型摘要已导出到: {output_file}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

def cleanup_old_models(models: List[Dict], keep_count: int = 3):
    """清理旧模型，保留最新的几个"""
    if len(models) <= keep_count:
        print(f"📁 模型数量 ({len(models)}) 不超过保留数量 ({keep_count})，无需清理")
        return
    
    print(f"\n🧹 清理旧模型 (保留最新的 {keep_count} 个)")
    
    models_to_keep = models[:keep_count]
    models_to_delete = models[keep_count:]
    
    print(f"📋 将保留的模型:")
    for model in models_to_keep:
        print(f"   ✅ {model['filename']}")
    
    print(f"\n🗑️  将删除的模型:")
    for model in models_to_delete:
        print(f"   ❌ {model['filename']}")
    
    # 确认删除
    confirm = input(f"\n确认删除 {len(models_to_delete)} 个旧模型? (y/N): ").strip().lower()
    if confirm == 'y':
        deleted_count = 0
        for model in models_to_delete:
            try:
                # 删除主模型文件
                os.remove(model['path'])
                
                # 删除相关文件
                onnx_path = model['path'].replace('.pth', '.onnx')
                info_path = model['path'].replace('.pth', '_info.json')
                
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
                if os.path.exists(info_path):
                    os.remove(info_path)
                
                deleted_count += 1
                print(f"   🗑️  已删除: {model['filename']}")
                
            except Exception as e:
                print(f"   ❌ 删除失败 {model['filename']}: {e}")
        
        print(f"\n✅ 清理完成，删除了 {deleted_count} 个模型")
    else:
        print("❌ 取消清理操作")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DDQN模型管理工具")
    parser.add_argument('--list', action='store_true', help='列出所有模型')
    parser.add_argument('--info', type=str, help='显示指定模型的详细信息')
    parser.add_argument('--compare', action='store_true', help='比较所有模型')
    parser.add_argument('--export', type=str, help='导出模型摘要到指定文件')
    parser.add_argument('--cleanup', type=int, metavar='N', help='清理旧模型，保留最新的N个')
    parser.add_argument('--model-dir', type=str, default='models', help='模型目录路径')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认列出模型
    if not any([args.list, args.info, args.compare, args.export, args.cleanup]):
        args.list = True
    
    # 列出模型
    models = list_models(args.model_dir)
    
    if not models:
        print("❌ 未找到任何模型文件")
        return
    
    print(f"📁 找到 {len(models)} 个模型文件")
    
    # 执行相应操作
    if args.list:
        print(f"\n📋 模型列表:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['filename']} ({model['size_mb']:.1f} MB)")
    
    if args.info:
        # 查找指定模型
        target_model = None
        for model in models:
            if args.info in model['filename']:
                target_model = model
                break
        
        if target_model:
            display_model_info(target_model)
        else:
            print(f"❌ 未找到包含 '{args.info}' 的模型")
    
    if args.compare:
        compare_models(models)
    
    if args.export:
        export_model_summary(models, args.export)
    
    if args.cleanup:
        cleanup_old_models(models, args.cleanup)

if __name__ == "__main__":
    main()
