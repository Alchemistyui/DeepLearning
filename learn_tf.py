import tensorflow as tf

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