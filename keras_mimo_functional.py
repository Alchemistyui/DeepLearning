
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
#main_data
#主数据集为10000*100的二维数组，意味着100个特征
#标签为10000*1的二维数组，共有10种输出结果
main_x = np.random.random((10000,100))
main_y = keras.utils.to_categorical(np.random.randint(10,size = (10000,1)))

# 主数据集和额外的数据集的输入的特征张量的数据集个数相等，也就是行数相等；
add_x = np.random.random((10000,10))
add_y = keras.utils.to_categorical(np.random.randint(10,size = (10000,1)))

# 设定主要输入的张量，并命名main_input
# input里shape=(特征个数,)
main_input = Input(shape=(100,), dtype='int32', name='main_input')
# Embedding在多输入多输入神经网络模型中的运用，只能作为模型的第一层
# 嵌入层将这个张量转化为一个 100*512的二维张量
# 主要目的有三：减小输入数据的维度（相对于one-hot encoding），提高运算效率，
# 可以自然实现词与词的相似度在矢量空间的表示。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
#print(x.shape)

#LSTM(Long Short Term Memory)长短期记忆模型是一种特殊的循环神经网络，表现得比标准的RNN要出色
lstm_out = LSTM(32)(x)
# 插入一个额外的损失，使得即使在主损失很高的情况下，LSTM和Embedding层也可以平滑的训练
auxiliary_output = Dense(10, activation='sigmoid', name='aux_output')(lstm_out)

#额外的输入数据
auxiliary_input = Input(shape=(10,), name='aux_input')

#将LSTM得到的张量与额外输入的数据串联起来，横向连接
x = keras.layers.concatenate([lstm_out, auxiliary_input])
#建立一个深层连接的网络
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#得到主数据集输出的张量，与输入的主数据集标签数据集的标签类相等
main_output = Dense(10, activation='softmax', name='main_output')(x)

# 整个2输入，2输出的模型
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
# 给额外的损失赋0.2的权重。我们可以通过关键字参数loss_weights或loss来为不同的输出设置不同的损失函数或权值
# 此处给loss传递单个损失函数，这个损失函数会被应用于所有输出上
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',loss_weights=[1., 0.2])
# 传递训练数据和目标值训练该模型
model.fit([main_x, add_x], [main_y, main_y],epochs=10, batch_size=128)



