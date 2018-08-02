import tensorflow as tf

# The master places operations on worker nodes and executes them 
# several times. On completion the master signals the workders
# to shut down. 

host = "127.0.0.1:"
port1 = 9500 # port of server 1
port2 = 9501 # port of server 2
cluster = {"worker": [host+str(port1), host+str(port2)]}
clusterspec = tf.train.ClusterSpec(cluster).as_cluster_def()

sess = tf.Session("grpc://"+host+str(port1)) # does not matter which worker we connect to
queue0 = tf.FIFOQueue(1, tf.int32, shared_name="queue0")
queue1 = tf.FIFOQueue(1, tf.int32, shared_name="queue1")

# define some operations and assign them to workers 
with tf.device("/job:worker/task:0"):
  add_op0 = tf.add(tf.fill((), 2.0), tf.fill((), 3.0))
with tf.device("/job:worker/task:1"):
  add_op1 = tf.add(tf.fill((), 4.0), tf.fill((), 5.0))

# run the operations
for i in range(5):
    print('Iteration: %d' % i)
    print("Running computation on server 0")
    print(sess.run(add_op0))
    print("Running computation on server 1")
    print(sess.run(add_op1))
      
print("Bringing down server 0")
sess.run(queue0.enqueue(1))
print("Bringing down server 1")
sess.run(queue1.enqueue(1))