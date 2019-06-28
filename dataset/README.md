
# tf.data.Dataset examples

## Standard pipline example

The standard pipeline example looks like this: 

    def input_fn(batch_size):
        files = tf.data.Dataset.list_files(file_pattern)
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=20)
        ds = ds.shuffle(buffer_size=10000)
        ds = ds.repeat(NUM_EPOCHS)
        ds = ds.map(parser_fn, num_parallel_calls=20)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=10)
        return ds

There are fused functions in `tf.contrib` to implement an optimized version:

    def input_fn(batch_size):
        files = tf.data.Dataset.list_files(file_pattern)
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=20)
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000, NUM_EPOCHS)
        ds = ds.apply(tf.contrib.data.map_and_batch(parser_fn, batch_size))
        ds = ds.prefetch(buffer_size=10)
        return ds


## Fused pipeline operators

Look at experimental Dataset functions in [tf.contrib.data](https://www.tensorflow.org/api_docs/python/tf/contrib/data).
There package contains several `fused pipeline operators` like:

 * [tf.contrib.data.shuffle_and_repeat](https://www.tensorflow.org/api_docs/python/tf/contrib/data/shuffle_and_repeat)
 * [tf.contrib.data.map_and_batch](https://www.tensorflow.org/api_docs/python/tf/contrib/data/map_and_batch)

 ## Advanced techniques 

 * enable sloppy interleave
 * adjust read buffer sizes
 * perform transformation on accelerators
 
