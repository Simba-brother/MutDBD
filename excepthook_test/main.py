import logging
import sys

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# 自定义异常钩子
def my_excepthook(exctype, value, traceback):
    logging.critical("Uncaught exception", exc_info=(exctype, value, traceback))
    # 调用默认的异常钩子，以防程序意外退出
    sys.__excepthook__(exctype, value, traceback)

# 设置新的异常钩子，即系统发生异常时会自动调用钩子函数（my_excepthook）钩子中含有logging
sys.excepthook = my_excepthook
def f1(a,b):
    print(b)

f1("mml")
