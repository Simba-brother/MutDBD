import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
# print(tf.__version__)
# print(tf.__file__)
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential

def build_dataset():
    # 从keras内置模型库中导入MNIST数据
    (X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
    X_train = X_train.reshape((60000, 28, 28, 1))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.reshape((10000, 28, 28, 1))
    X_test = X_test.astype('float32') / 255

    #将label映射为one-hot的形式
    num_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    return (X_train, Y_train),(X_test, Y_test)

def build_model():
    model = Sequential()
    # 卷积层
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding = 'same',input_shape = (28,28,1),activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    # 卷积层
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding = 'same',input_shape = (14,14,16),activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    # 打平
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))#建立输出层
    # 输出层
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def start_train():
    (X_train, Y_train),(X_test, Y_test) = build_dataset()
    model = build_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="code/output")
    model.fit(x=X_train, y=Y_train, batch_size=32, epochs=10, 
              validation_split = 0.2,
              # validation_data=(X_test, Y_test), 
              verbose=2, # 日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
              callbacks=[tensorboard_callback])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    start_train()