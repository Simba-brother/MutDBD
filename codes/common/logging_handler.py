import logging
import time
import os
def get_logging(log_file_dir:str,log_file_name:str,filemode:str):
    # 日志文件目录
    os.makedirs(log_file_dir,exist_ok=True)
    # 日志文件名
    log_file_path = os.path.join(log_file_dir,log_file_name)
    # 日志格式
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=log_file_path,filemode=filemode)
    return logging

if __name__ == "__main__":
    get_logging()