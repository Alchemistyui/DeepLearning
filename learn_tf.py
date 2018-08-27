import tensorflow as tf
# import numpy as np

# 此时只是创建了一个graph，而并不会真正的输出结果
a = tf.add(3, 5)


# 创建一个session并在其里面运行graph
sess = tf.Session()
# run里可以放一整个计算图，现在是放了一个节点
print(sess.run(a))
sess.close()

# -----------------

x = 2
y = 3
# tf是反向计算，不会计算无用的结点
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
useless = tf.add(x, op1)
op3 = tf.pow(op2, op1)

# 另一种创建session的方法
with tf.Session() as sess:
    print(sess.run(op3))
    # 有多个fetch节点参数时用list运行多任务
    # print(sess.run([op3, useless]))


# tf也能让计算图的各个子图并行，可分配给不同特定GPU

# 在特定GPU上创建graph,实例化所有结点
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0], name='b')
    # c = tf.matmul(a, b)

# 。。。。


# tf一般是在默认图里运行，即sess.run()，但有时想创建多个图，如对同一个任务，想同时训练好几个模型
# 为了防止混用，最好先保存下默认图方便切换 
# 获得默认图的句柄
g1 = tf.get_default_graph()
# 要向一个图添加结点，得先把其设置为默认图
g = tf.Graph()
with g.as_default():
    x = tf.add(3, 5)

# sess = tf.Session(graph = g)
with tf.Session(graph = g) as sess:
    sess.run(x)


# 建立多个图的好处是可以随意执行某个子图，而不是全图，而子图可以只执行某一op 
# 同时也能将计算分解，方便求自动差分，和对分布式计算有帮助


