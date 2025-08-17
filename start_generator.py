# 导入消息生成器模块
from emulator import message_generator

def main():
    # 启动消息生成器，设置最小间隔为1秒，最大间隔为2秒
    message_generator.start(min_intv=1, max_intv=2)

# 如果直接运行此脚本，则执行main函数
if __name__ == '__main__':
    main()