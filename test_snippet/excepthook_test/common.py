import logging
import sys
# 自定义异常钩子
def my_excepthook(exctype, value, traceback):
    logging.critical("Uncaught exception", exc_info=(exctype, value, traceback))
    # 调用默认的异常钩子，以防程序意外退出
    sys.__excepthook__(exctype, value, traceback)