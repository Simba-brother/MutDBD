import logging
import sys
from excepthook_test.common import my_excepthook
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


# 设置新的异常钩子，即系统发生异常时会自动调用钩子函数（my_excepthook）钩子中含有logging
sys.excepthook = my_excepthook
def f1(a,b):
    logging.info(f"faf{a}")
    print(b)

f1(1,2)
