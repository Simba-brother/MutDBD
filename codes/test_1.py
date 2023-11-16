import torch
import torchvision
import onnx
import onnx_tf
from onnx_tf.backend import prepare
import tensorflow as tf
from onnx2keras import onnx_to_keras
# https://blog.csdn.net/weixin_44034578/article/details/120947140 


def pth_to_onnx():
  '''
  pth model to onnx model
  '''
  model_pth = torchvision.models.resnet18(pretrained=True)
  torch.onnx.export(model_pth,               # model being run
                    torch.randn(1, 3, 224, 224), # dummy input (required)
                    "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,
                    input_names=["input"],        # 输入名
                    output_names=["output"]    # 输出名
                  ) # store the trained parameter weights inside the model file

def onnx_to_keras():

  # Load the ONNX model
  model_onnx = onnx.load("resnet18.onnx")
  '''
  # Check that the model is well-formed
  onnx.checker.check_model(model_onnx)
  '''
  k_model = onnx_to_keras(model_onnx, ['input'])
  # tf.keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)    #第二个参数是新的.h5模型的保存地址及文件名
  k_model.summary()

if __name__ == "__main__":
    pass

    