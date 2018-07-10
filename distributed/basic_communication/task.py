import argparse
import tensorflow as tf

# The worker runs computations invoked by the master. It
# terminates after receiving a signal from the master.

parser = argparse.ArgumentParser()
parser.add_argument('task_id', help='Task ID', type=int)
args = parser.parse_args()    

host = "127.0.0.1:"
port1 = 9500 # port of server 1
port2 = 9501 # port of server 2
cluster = {"worker": [host+str(port1), host+str(port2)]}
clusterspec = tf.train.ClusterSpec(cluster).as_cluster_def()

task = args.task_id
server = tf.train.Server(clusterspec, config=None,
                         job_name="worker",
                         task_index=task)
print("Starting server %d" % task)
sess = tf.Session(server.target)
queue = tf.FIFOQueue(1, tf.int32, shared_name="queue%d" % task)
sess.run(queue.dequeue())
print("Terminating server %d" % task)