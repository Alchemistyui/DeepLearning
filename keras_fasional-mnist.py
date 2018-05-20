from __future__ import print_function
import keras
# from keras.datasets import mnist
from keras.datasets import fashion_mnist
# import mnist_reader
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras import backend as K


# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。
batch_size = 128
# 0-9手写数字一个有10个类别
num_classes = 10
# epochs,12次完整迭代
epochs = 12
# 输入的图片是28*28像素的灰度图
img_rows, img_cols = 28, 28
# 训练集，测试集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 二维数据变成一维数据
# x_train = x_train.reshape(len(x_train), -1)
# x_test = x_test.reshape(len(x_test), -1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# uint不能有负数，先转为float类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 数据归一化
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 把类别0-9变成2进制，方便训练
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


# # This returns a tensor
# inputs = Input(shape=(784,))
inputs = Input(shape=(28, 28, 1))
# input_shape = (img_rows, img_cols, 1)




x = Conv2D(32,(3, 3), activation="relu", padding='same')(inputs)
# 64个通道的卷积层
x = Conv2D(64, (3, 3), activation="relu")(x)
# 池化层是2*2像素的
x = MaxPooling2D(pool_size=(2, 2))(x)
# 对于池化层的输出，采用0.2概率的Dropout
x = Dropout(0.2)(x)
# 展平所有像素，比如[28*28] -> [784]
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
x = Flatten()(x)
# 对所有像素使用全连接层，输出为128，激活函数选用relu
x = Dense(128, activation='relu')(x)
# 对输入采用0.5概率的Dropout
x = Dropout(0.2)(x)
# 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
predictions = Dense(num_classes, activation='softmax')(x)




# a layer instance is callable on a tensor, and returns a tensor
# x = Dense(784, activation='relu', 
#     kernel_initializer='he_normal')(inputs)
# x = Dropout(0.2)(x)
# x = Dense(512, activation='relu', 
#     kernel_initializer='he_normal')(x)
# x = Dropout(0.2)(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
 verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])