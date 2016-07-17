import cPickle as pickle
import tensorflow as tf
import numpy as np

image_size = 28
num_labels = 10

with open('name_of_pickled_file','rb') as f :
    file = pickle.load(f)
    train_dataset = file['train_dataset']
    train_labels = file['train_labels']
    valid_dataset =file['valid_dataset']
    valid_labels= file['valid_labels']
    test_dataset = file['test_dataset']
    test_labels = file['test_labels']
    del file
"""
convert the dataset from 3d (n,img_size,img_size) to 2d (n,img_size*img_size)
and labels vactors map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
"""
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

"""
Build a graph for the network built using tensorflow's core graph functions
and load the input data
"""
batch_size = 128

graph = tf.Graph()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with graph.as_default():

  # Input data. For the training data use a placeholder that will be fed at run time
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_test_dataset = tf.constant(test_dataset)
  tf_valid_dataset =tf.constant(valid_dataset)

  # Variables.
  hl_weights = weight_variable([image_size * image_size, 1024])
  hl_biases = bias_variable([1024])
  weights =weight_variable([1024,num_labels])
  biases =bias_variable([num_labels])

  input_layer= tf.matmul(tf_train_dataset,hl_weights) + hl_biases
  hidden_layer =tf.nn.relu(input_layer)
  last_layer = tf.matmul(hidden_layer,weights) + biases

  # Training computation.
  y_out =tf.nn.softmax(last_layer) #network output =softmax last_layer
  loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y_out, tf_train_labels) )


  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training and test data.
  train_prediction = y_out
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,hl_weights) + hl_biases),weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,hl_weights) + hl_biases),weights) + biases)

#Training using stochastic gradient descent
num_steps = 3001
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
"""dictionary prepared and filled during the session"""
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
