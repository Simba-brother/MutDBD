import logging
import time
import os
def get_Logger(logger_name:str,log_file_dir:str,log_file_name:str,filemode:str):
    '''
    filemode:"a"|"w"
    '''
    # 日志者
    logger =logging.Logger(logger_name)

    # 文件处理者
    os.makedirs(log_file_dir,exist_ok=True)
    log_file_path = os.path.join(log_file_dir,log_file_name)
    fileHandler = logging.FileHandler(log_file_path,mode=filemode)
    # 格式者
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    formatter = logging.Formatter(LOG_FORMAT)

    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    

    logger.addHandler(fileHandler)
    return logger

if __name__ == "__main__":
    log_file_dir = "log/test"
    logger_1 = get_Logger("mml",log_file_dir,log_file_name="log1.log",filemode="w")
    logger_1.debug("logging_1_content")
    logger_2 = get_Logger("mml",log_file_dir,log_file_name="log2.log",filemode="w")
    logger_2.debug("logging_2_content")
    