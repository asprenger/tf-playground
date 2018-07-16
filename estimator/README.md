
This examples are about non-distributed training on single GPU:

 * [mnist_basic.py](mnist_basic.py) - Build and train a classifier using the low-level API
 * [mnist_estimator.py](mnist_estimator.py) - Build and train a classifier using the Estimator API
 * [mnist_keras.py](mnist_keras.py) - Build a Keras model and train it with the Estimator API
 * [mnist_train_and_evaluate.py](mnist_train_and_evaluate.py) - Build and train a classifier using the train_and_evaluate() function

# Links

 * https://www.tensorflow.org/api_docs/python/tf/train/Supervisor
 * https://www.tensorflow.org/api_guides/python/train#Distributed_execution
 * https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
 * https://www.tensorflow.org/versions/r1.0/programmers_guide/supervisor
 * https://www.tensorflow.org/versions/r1.0/programmers_guide/threading_and_queues
 * https://www.tensorflow.org/versions/r1.0/programmers_guide/reading_data
 * https://www.tensorflow.org/versions/r1.0/programmers_guide/variables

# Available SessionRunHooks

SessionRunHooks allow to intercept the session lifecycle at the following points:

 * when a session starts being used
 * before a call to the `session.run()`
 * after a call to the `session.run()`
 * when the session closed

For more details about the API check out [tensorflow/python/training/session_run_hook.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/session_run_hook.py).

List of available SessionRunHooks:

 * tf.train.LoggingTensorHook: Prints a given set of tensors every N local steps or every N seconds.
 * tf.train.StopAtStepHook: Hook that requests stop at a specified step.
 * tf.train.CheckpointSaverHook: Saves checkpoints every N steps or seconds.
 * tf.train.StepCounterHook: Hook that counts and logs steps per second.
 * tf.train.NanTensorHook: Hook that stops training if loss is NaN.
 * tf.train.SummarySaverHook: Saves summaries every N steps.
 * tf.train.GlobalStepWaiterHook: Delays execution until global step reaches a given value. This is used to gradually start workers in distributed settings.
 * tf.train.FinalOpsHook: A hook that evaluates tensors at the end of a session.
 * tf.train.FeedFnHook: Overwrite the `feed_dict` with a function result
 * tf.train.ProfilerHook: Captures CPU/GPU profiling information every N steps or seconds.

# Distributed training with tf.estimator.train_and_evaluate

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

