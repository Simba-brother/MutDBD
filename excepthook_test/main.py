import logging
import sys
from excepthook_test.common import my_excepthook


# 设置新的异常钩子，即系统发生异常时会自动调用钩子函数（my_excepthook）钩子中含有logging
sys.excepthook = my_excepthook



def f1(a,b):
    
    logger.debug(f"faf{a}")
    print(b)




for i in range(2):
    logger = logging.getLogger(f"{i}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f'excepthook_test/test_{i}.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    f1(i,i)