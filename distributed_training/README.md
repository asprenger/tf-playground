
# Distributed training

## Relevant TensorFlow source files

 * [estimator.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/estimator.py):
    * tf.estimator.Estimator

 * [training.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/training.py):
    * tf.estimator.train_and_evaluate

 * [monitored_session.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/monitored_session.py):
    * tf.train.MonitoredTrainingSession
    * tf.train.MonitoredSession

## Low level distributed training

[mnist_low_level_distributed.py](mnist_low_level_distributed.py) implements distributed training with *data parallelism* 
and *asynchronous parameter updates*. In TensorFlow terminology data parallelism is called *between-graph replication*. 
The script implements the approach described in [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

To run the example on a local machine with a parameter server and two workers execute the following commands:

    python mnist_low_level_distributed.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name=ps --task_index=0
    python mnist_low_level_distributed.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name=worker --task_index=0
    python mnist_low_level_distributed.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name=worker --task_index=1
    
The checkpoint directory must be on a shared filesystem and accessible to all processes in the cluster.

Parameter servers aggregate gradients and broadcast updates, they are typically placed on CPU devices. Workers 
calculate gradients during training, they are typically placed on a GPU devices. The worker with index 0 is called 
the chief worker. The chief worker: 

 * coordinates model training
 * initializes the model
 * counts the number of training steps completed
 * monitors the session
 * saves logs for TensorBoard
 * saves and restores model checkpoints to recover from failures. 

The chief worker also manages failures, ensuring fault tolerance if a worker or parameter server fails. If a worker 
fails it will contact the parameter servers and continue as before because a worker is effectively stateless. If a 
parameter server dies, the chief worker recovers from the last checkpoint after a new parameter server joins the 
system. If the chief worker itself dies, training will need to be restarted from the most recent checkpoint. Most of 
the functionality is provided by [tf.train.MonitoredTrainingSession](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession) 
and [tf.train.MonitoredSession](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession).


## Using tf.estimator.train_and_evaluate

[mnist_train_and_evaluate.py](mnist_train_and_evaluate.py) creates an estimator and calls `tf.estimator.train_and_evaluate`. 
This helper function coordinates the training and evaluation of the model. The function provides consistent behavior for both 
local and distributed training. It only supports data parallelism (between-graph replication) and asynchronous parameter 
update for the distributed configuration.

### Local training

Run local training:

    python mnist_train_and_evaluate.py

### Distributed training

Start parameter server:

    TF_CONFIG='{
        "cluster": {
            "chief": ["localhost:2000"],
            "ps": ["localhost:2002"],
            "worker": ["localhost:2001"]
        },
        "task": {"type": "ps", "index": 0}
    }' python mnist_train_and_evaluate.py

Start chief worker:

    TF_CONFIG='{
        "cluster": {
            "chief": ["localhost:2000"],
            "ps": ["localhost:2002"],
            "worker": ["localhost:2001"]
        },
        "task": {"type": "chief", "index": 0}
    }' python mnist_train_and_evaluate.py

Start worker:

    TF_CONFIG='{
        "cluster": {
            "chief": ["localhost:2000"],
            "ps": ["localhost:2002"],
            "worker": ["localhost:2001"]
        },
        "task": {"type": "worker", "index": 0}
    }' python mnist_train_and_evaluate.py

Start evaluator:

    TF_CONFIG='{
        "cluster": {
            "chief": ["localhost:2000"],
            "ps": ["localhost:2002"],
            "worker": ["localhost:2001"]
        },
        "task": {"type": "evaluator", "index": 0}
    }' python mnist_train_and_evaluate.py


# Process management

TensorFlow provides only the basic functionality that is necessary for distributed training. The problem is that you 
have to manage the starting and stopping of processes yourself. This means keeping track of the IP addresses and 
ports of all TensorFlow servers in your cluster, and starting and stopping those servers manually. To do this in an
automatic fashion you will probably use a resource management framework like:

 * YARN
 * Kubernetes
 * Mesos

Another option to manage process lifecycles would be Spark. The advantage of Spark is that it is able to run on the
mentioned resource management frameworks and serves as a kind of abstraction layer.

 
