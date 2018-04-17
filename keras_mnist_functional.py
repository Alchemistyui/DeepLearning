from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 二维数据变成一维数据
x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)


# uint不能有负数，先转为float类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 数据归一化,减去均值除以范围,最终是0-1的范围,
# 所以最后的激活函数应该是sigmoid,如果是-1~1,那么激活函数应该是tanh
x_train /= 255
x_test /= 255

# 把类别0-9变成2进制，方便训练
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(784, activation='relu', kernel_initializer='he_normal')(inputs)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
 verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])