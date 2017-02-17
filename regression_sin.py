import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


def data(step_length):
    x = np.arange(0,2*np.pi,step_length)
    x = np.reshape(x,[len(x),1])
    y = np.sin(x)

    return x,y

def plot_results(test_x,test_y,predicted_y):
    true_val, =plt.plot(test_x,test_y, 'b--', label="true")
    prediction, = plt.plot(test_x,predicted_y,'r^', label="predicted")
    plt.title("Learning sin(x) with 1 hidden layer with 5 nodes (Relu activation)")
    plt.legend()
    plt.show()


def plot_results_1_layer_width(test_x,test_y,predicted_y_arr,width_array):
    colors = ['c','g','r','m']
    plt.figure()
    true_val, =plt.plot(test_x,test_y, 'b--', label="true")
    for  i in np.arange(0,len(predicted_y_arr),1):
        prediction, = plt.plot(test_x,predicted_y_arr[i], label="%d nodes"%(width_array[i]))

    plt.title("Predicting sin(x) with 1 hidden layers of different widths")
    plt.legend()
    plt.show()


def learn_1_hidden_layer():

    node_count_hl1 = 5

    x = tf.placeholder(tf.float32,[None,1])
    y = tf.placeholder(tf.float32,[None,1])

    hidden_layer_1 = {
    "weights": tf.Variable(tf.random_normal([1,node_count_hl1])),
    "biases": tf.Variable(tf.random_normal([node_count_hl1]))
    }

    output_layer = {
    "weights": tf.Variable(tf.random_normal([node_count_hl1,1])),
    "biases": tf.Variable(tf.random_normal([1]))
    }

    hl1_output = tf.nn.relu(tf.add(tf.matmul(x,hidden_layer_1["weights"]),hidden_layer_1["biases"]))

    predicted_y = tf.add(tf.matmul(hl1_output,output_layer["weights"]),output_layer['biases'])

    mse = tf.reduce_mean(tf.square(tf.sub(y,predicted_y)))

    opt = tf.train.AdamOptimizer()
    opt_operation = opt.minimize(mse)

    train_x,train_y = data(0.001)

    init_op = tf.global_variables_initializer()

    test_x,test_y = data(0.1)

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(1000):
            _,loss_val =  sess.run([opt_operation,mse],feed_dict={x: train_x, y: train_y})

        # Training is done, use the trained weights to get the predictions
        predicted_y_numpy = sess.run(predicted_y,feed_dict={x: test_x})

        plot_results(test_x,test_y,predicted_y_numpy)

def learn_1_hidden_layer_wider():

    test_x,test_y = data(0.1)
    train_x,train_y = data(0.001)

    # This will hold the predictions by all the neural nets
    predicted_y_numpy_arr = []

    width_array = [5,10,15,20,25,30]
    for node_count_hl1 in width_array :
        hidden_layer_1 = {
            "weights": tf.Variable(tf.random_normal([1,node_count_hl1])),
            "biases": tf.Variable(tf.random_normal([node_count_hl1]))
        }

        output_layer = {
            "weights": tf.Variable(tf.random_normal([node_count_hl1,1])),
            "biases": tf.Variable(tf.random_normal([1]))
        }

        x = tf.placeholder(tf.float32,[None,1])
        y = tf.placeholder(tf.float32,[None,1])

        hl1_output = tf.nn.relu(tf.add(tf.matmul(x,hidden_layer_1["weights"]),hidden_layer_1["biases"]))

        predicted_y = tf.add(tf.matmul(hl1_output,output_layer["weights"]),output_layer['biases'])

        mse = tf.reduce_mean(tf.square(tf.sub(y,predicted_y)))

        opt = tf.train.AdamOptimizer()
        opt_operation = opt.minimize(mse)


        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in range(10000):
                _,loss_val =  sess.run([opt_operation,mse],feed_dict={x: train_x, y: train_y})

            # Training is done, use the trained weights to get the predictions
            predicted_y_numpy = sess.run(predicted_y,feed_dict={x: test_x})
            predicted_y_numpy_arr.append(predicted_y_numpy)

        
        plot_results_1_layer_width(test_x,test_y,predicted_y_numpy_arr,width_array)

def learn_2_hidden_layers():

    node_count_hl1 = 5

    x = tf.placeholder(tf.float32,[None,1])
    y = tf.placeholder(tf.float32,[None,1])

    hidden_layer_1 = {
    "weights": tf.Variable(tf.random_normal([1,node_count_hl1])),
    "biases": tf.Variable(tf.random_normal([node_count_hl1]))
    }

    hidden_layer_2
    output_layer = {
    "weights": tf.Variable(tf.random_normal([node_count_hl1,1])),
    "biases": tf.Variable(tf.random_normal([1]))
    }

    hl1_output = tf.nn.relu(tf.add(tf.matmul(x,hidden_layer_1["weights"]),hidden_layer_1["biases"]))

    predicted_y = tf.add(tf.matmul(hl1_output,output_layer["weights"]),output_layer['biases'])

    mse = tf.reduce_mean(tf.square(tf.sub(y,predicted_y)))

    opt = tf.train.AdamOptimizer()
    opt_operation = opt.minimize(mse)

    train_x,train_y = data(0.001)

    init_op = tf.global_variables_initializer()

    test_x,test_y = data(0.1)

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(1000):
            _,loss_val =  sess.run([opt_operation,mse],feed_dict={x: train_x, y: train_y})

        # Training is done, use the trained weights to get the predictions
        predicted_y_numpy = sess.run(predicted_y,feed_dict={x: test_x})

        plot_results(test_x,test_y,predicted_y_numpy)

learn_1_hidden_layer_wider()