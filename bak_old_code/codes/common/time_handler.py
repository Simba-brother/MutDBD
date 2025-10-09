import time
def get_formattedDateTime():
    '''
    用于生成格式化的时间
    '''
    timestamp = time.time()
    date_time = time.localtime(timestamp)
    formatted_time = time.strftime('%Y-%m-%d_%H:%M:%S', date_time)
    return formatted_time

if __name__ == "__main__":
    _t = get_formattedDateTime()
    print(_t)