import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

ema_op = ema.apply([update])
with tf.control_dependencies([ema_op]):
    ema_val = tf.identity(ema.average(update))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(3):
    print(sess.run([ema_val]))
sess.close()
