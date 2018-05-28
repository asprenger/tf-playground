import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def build_model(x):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return logits, keep_prob

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def main(_):

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name, # 'ps' or 'worker'
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()

  elif FLAGS.job_name == "worker":

    # The device function assigns each node in the graph to a device.
    device_fn = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)

    with tf.device(device_fn):

      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

      x = tf.placeholder(tf.float32, [None, 784])
      y = tf.placeholder(tf.float32, [None, 10])
      logits, keep_prob = build_model(x)

      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) # TODO compare with mnist_basic.py

      global_step = tf.train.get_or_create_global_step()

      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=1000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when 
    # done or an error occurs. 
    # `server.target` returns a URL for a tf.Session to connect to this server.
    # The master worker has the task_index 0.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.log_dir,
                                           hooks=hooks) as mon_sess:

      i = 0 # TODO rename with 'step'
      while not mon_sess.should_stop():
        
        batch = mnist.train.next_batch(50)

        if i % 50 == 0:
          feed_dict = {
              x: batch[0], 
              y: batch[1], 
              keep_prob: 1.0
          }
          train_accuracy = mon_sess.run(accuracy, feed_dict=feed_dict)
          print('global_step %s, task:%d_step %d, training accuracy %g' % (tf.train.global_step(mon_sess, global_step), FLAGS.task_index, i, train_accuracy))

        # Run a training step asynchronously.
        feed_dict = {
          x: batch[0], 
          y: batch[1], 
          keep_prob: 0.5 
        }
        mon_sess.run(train_step, feed_dict=feed_dict)
        i = i + 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )

  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )

  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/tensorflow/mnist",
      help="Directory for storing input data")

  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/train_logs",
      help="Directory for train logs")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
