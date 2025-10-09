class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg): # 实例() 则该方法会被调用
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)
