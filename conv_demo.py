import time
import load
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Import DataSet and One_Hot labels
mnist_fashion = load.read_data_sets(data_dir='./data',one_hot= True)

n_classes = 10
label_names = {0:'T-shirt.top',
               1:'Trouser',
               2:'Pullover',
               3:'Dress',
               4:'Coat',
               5:'Sandal',
               6:'Shirt',
               7:'Sneaker',
               8:'Bag',
               9:'Ankle-boot'}

train_X = mnist_fashion.train.images.reshape(-1, 28, 28, 1)
test_X = mnist_fashion.test.images.reshape(-1,28,28,1)

train_y = mnist_fashion.train.labels
test_y = mnist_fashion.test.labels

#Hyperparameter
training_iters = 10
learning_rate = 0.001
batch_size = 128
#dropout = 0.75

x = tf.placeholder('float', [None, 28,28,1])
y = tf.placeholder('float', [None, n_classes])
#keep_prob = tf.placeholder(tf.float32)

# Conv2D layer, with bias and relu activation
def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

#Max pooling
def pooling(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
    'w1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    'w2': tf.Variable(tf.random_normal([3, 3, 32, 64])), 
    'w3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wf': tf.Variable(tf.random_normal([4*4*128, 128])), 
    'out': tf.Variable(tf.random_normal([128, n_classes])),
}

biases = {
    'b1': tf.Variable(tf.random_normal([32])),
    'b2': tf.Variable(tf.random_normal([64])),
    'b3': tf.Variable(tf.random_normal([128])),
    'bf': tf.Variable(tf.random_normal([128])),
    'out': tf.Variable(tf.random_normal([128, n_classes])),
}

# Convolution Layer
def conv_net(x, weights, biases): 
    #here we call the conv2d function we had defined above and pass the input image x, weights w1 and bias b1.
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    #Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window.
    conv1 = pooling(conv1, k=2)
    #conv1 = tf.nn.dropout(conv1,keep_prob)

    
    #Similar to conv1.
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    conv2 = pooling(conv2, k=2)
    #conv2 = tf.nn.dropout(conv2,keep_prob)

    conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    conv3 = pooling(conv3, k=2)
    #conv3 = tf.nn.dropout(conv3,keep_prob)
    

    #Fully connected layer
    #Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wf'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wf']), biases['bf'])
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1,keep_prob)

    #Output, class prediction
    #Finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases) 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted is equal to the actual label.
#Both of them are 1-Dimension vector.
#For example: if one be predicted to 'Dress',the corresponing vector is [0,0,0,0,1,0,0,0,0,0]
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init =tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    print('Training begin at:',time.strftime('%c',time.localtime()))

    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            #Here should run dropout to avoid OVERFITTING.
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print('----------------------------------------------------------------')
        print('Iterative times :',str(i),'Loss ={:.6f}'.format(loss),\
                      'Training Accuracy= {:.5f}'.format(acc))
        print("ONE Time Iteration Finished!")

        # Calculate accuracy for one batch test images
        test_acc = sess.run(accuracy, feed_dict={x: test_X[:128],y : test_y[:128]})
        valid_loss = sess.run(cost, feed_dict={x: test_X[:128], y : test_y[:128]})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print('Testing Accuracy:{:.5f}'.format(test_acc))
        print('Iteration END:',time.strftime('%c',time.localtime()))
        print('----------------------------------------------------------------')
    print('Training done at:',time.strftime('%c',time.localtime()))