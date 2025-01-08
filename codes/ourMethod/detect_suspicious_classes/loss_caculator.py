from sklearn.metrics import log_loss
import pandas as pd
def get_cross_entropy_loss(y_true,y_pred,labels):
    log_loss(y_true,y_pred,labels)


def main():
    # 加载confidence csv
    pass

if __name__ == "__main__":
    main()