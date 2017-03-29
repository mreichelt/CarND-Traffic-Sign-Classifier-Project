# Load pickled data
import pickle
import tensorflow as tf

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
#import matplotlib.pyplot as plt
import random
import numpy as np

samples_to_show = 1

indexes = list(range(n_train))
random.shuffle(indexes)
indexes = indexes[0:samples_to_show]
#for i in indexes:
    #imgplot = plt.imshow(X_train[i])
    # plt.show()

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def grayscale(X):
    # we simply add up the colors - they will be normalized away anyway later on
    return np.sum(X, axis=3, keepdims=True)

def feature_scaled(X, min, max):
    return (X - min) / (max - min)

print('applying grayscale')
X_train = grayscale(X_train)
X_valid = grayscale(X_valid)
X_test = grayscale(X_test)

print('applying feature scaling')
min = np.min([np.min(X_train), np.min(X_valid), np.min(X_test)])
max = np.max([np.max(X_train), np.max(X_valid), np.max(X_test)])
X_train = feature_scaled(X_train, min, max)
X_valid = feature_scaled(X_valid, min, max)
X_test = feature_scaled(X_test, min, max)


### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    out1 = 6
    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, out1], mu, sigma))
    b1 = tf.Variable(tf.zeros(out1))
    conv1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID') + b1

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 2: Convolutional. Output = 10x10x16.
    out2 = 16
    w2 = tf.Variable(tf.truncated_normal([5, 5, out1, out2], mu, sigma))
    b2 = tf.Variable(tf.zeros(out2))
    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='VALID') + b2

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Flatten. Input = 5x5x16. Output = 400.
    flat_out = 5 * 5 * out2
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    out3 = 120
    w3 = tf.Variable(tf.truncated_normal([flat_out, out3], mu, sigma))
    b3 = tf.Variable(tf.zeros(out3))
    fc1 = tf.matmul(fc0, w3) + b3

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    out4 = 84
    w4 = tf.Variable(tf.truncated_normal([out3, out4], mu, sigma))
    b4 = tf.Variable(tf.zeros(out4))
    fc2 = tf.matmul(fc1, w4) + b4

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43 (n_classes).
    w5 = tf.Variable(tf.truncated_normal([out4, n_classes], mu, sigma))
    b5 = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, w5) + b5

    return logits





### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

learning_rate = 0.001
batch_size = 128
epochs = 10

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("Epoch {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './model')
    print("Model saved")


### Load the images and plot them here.
### Feel free to use as many code cells as needed.




### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.





### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.






### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.






### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

# def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
#     # Here make sure to preprocess your image_input in a way your network expects
#     # with size, normalization, ect if needed
#     # image_input =
#     # Note: x should be the same name as your network's tensorflow data placeholder variable
#     # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
#     activation = tf_activation.eval(session=sess, feed_dict={x: image_input})
#     featuremaps = activation.shape[3]
#     plt.figure(plt_num, figsize=(15, 15))
#     for featuremap in range(featuremaps):
#         plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
#         plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
#         if activation_min != -1 & activation_max != -1:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
#                        vmax=activation_max, cmap="gray")
#         elif activation_max != -1:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
#         elif activation_min != -1:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
#         else:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")
