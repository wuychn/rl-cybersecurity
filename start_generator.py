from emulator import message_generator  # 导入消息生成器模块

def main():  # 主函数
    message_generator.start(min_intv=1, max_intv=2)  # 启动消息生成器，设置最小间隔1秒，最大间隔2秒

if __name__ == '__main__':  # 如果脚本直接运行（不是被导入）
    main()  # 调用主函数