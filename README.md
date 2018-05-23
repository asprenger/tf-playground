# tf-estimator
Exploration of TensorFlow estimator API

#placeholder = tf.get_default_graph().get_tensor_by_name(self.tensor_name + ':0')

for v in tf.all_variables():
    print(v)

#print(tf.get_default_graph().get_tensor_by_name('fc1' + ':0'))


print('-'*20)

foo = [n.name for n in tf.get_default_graph().as_graph_def().node]
for f in foo:
    print(f)



#for op in tf.get_default_graph().get_operations():
#    print str(op.name) 


def get_all_variables_with_name(var_name):
    name = var_name + ':0'
    return [var for var in tf.all_variables() if var.name.endswith(name)]

https://github.com/tensorflow/models/tree/master/official/mnist