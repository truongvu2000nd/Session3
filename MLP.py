import math
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

NUM_CLASSES = 20


class MLP:
    """
    A class used to represent a MLP network: Input -> Hidden -> Output
    """

    def __init__(self, vocab_size, hidden_size=50):
        self._vocab_size = vocab_size   # input size

        # create placeholders for tensorflow session
        self._X = tf.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_Y = tf.placeholder(tf.int32, shape=[None, 1])

        self._hidden_size = hidden_size     # number of units in the hidden layer

    def build_graph(self):
        """
        Build the computational graph: LINEAR -> RELU -> LINEAR -> SOFTMAX
        """
        weights1 = tf.get_variable(
            name="weights_input_hidden",
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.keras.initializers.glorot_uniform(seed=2018)  # Xavier initialization
        )
        biases1 = tf.get_variable(
            name="biases_input_hidden",
            shape=(1, self._hidden_size),
            initializer=tf.zeros_initializer()
        )

        weights2 = tf.get_variable(
            name="weights_hidden_output",
            shape=(self._hidden_size, NUM_CLASSES),
            initializer=tf.keras.initializers.glorot_uniform(seed=2018)  # Xavier initialization
        )
        biases2 = tf.get_variable(
            name="biases_hidden_output",
            shape=(1, NUM_CLASSES),
            initializer=tf.zeros_initializer()
        )

        # compute the loss
        # hidden = RELU(W1 * X + b1)
        # logits = W2 * hidden + b2
        # outputs = softmax(logits)
        # loss = cross_entropy(outputs, labels)

        hidden = tf.matmul(self._X, weights1) + biases1
        hidden = tf.nn.relu(hidden)
        logits = tf.matmul(hidden, weights2) + biases2

        labels_one_hot = tf.one_hot(indices=self._real_Y, depth=NUM_CLASSES, dtype=tf.float32)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)

        outputs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(outputs, axis=-1)

        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        # create an optimizer for tensorflow
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer


class DataReader:
    """
    A class used to read the data
    """
    def __init__(self, data_path, vocab_size, batch_size):
        self._batch_size = batch_size   # mini batch size
        self._data = []                 # input data
        self._labels = []               # real labels

        with open(data_path) as f:
            d_lines = f.readlines()

        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value

            self._data.append(vector)
            self._labels.append(label)

        self._data = np.array(self._data)                           # of shape (number of training examples, vocab_size)
        self._labels = np.array(self._labels).reshape(-1, 1)        # of shape (number of training examples, 1)

    def random_mini_batches(self, seed):
        """
        Randomly shuffles the data and labels
        :return:
        minibatches: A list of tuples, each tuple is a pair of (mini_batch data, mini_batch labels)
        """
        random.seed(seed)
        mini_batches = []
        m = self._data.shape[0]         # number of training examples

        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m / self._batch_size)

        # Step 1: Shuffle data and labels
        indices = list(range(m))
        random.shuffle(indices)
        shuffled_data, shuffled_labels = self._data[indices], self._labels[indices]

        # Step 2: Partition (shuffled_data, shuffled_labels). Minus the end case
        for k in range(0, num_complete_minibatches):
            mini_batch_data = shuffled_data[k * self._batch_size: k * self._batch_size + self._batch_size, :]
            mini_batch_labels = shuffled_labels[k * self._batch_size: k * self._batch_size + self._batch_size, :]
            mini_batch = (mini_batch_data, mini_batch_labels)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self._batch_size != 0:
            mini_batch_data = shuffled_data[num_complete_minibatches * self._batch_size: m, :]
            mini_batch_labels = shuffled_labels[num_complete_minibatches * self._batch_size: m, :]
            mini_batch = (mini_batch_data, mini_batch_labels)
            mini_batches.append(mini_batch)

        return mini_batches


def save_parameters(name, value, epoch):
    """Saves the parameters for future use."""
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1:  # is a list
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number)
                                           for number in value[row]])
                                 for row in range(value.shape[0])])

    with open('./saved_params/' + filename, 'w') as f:
        f.write(string_form)


def restore_parameters(name, epoch):
    """Restores saved parameters."""
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    with open('./saved_params/' + filename) as f:
        lines = f.readlines()
    if len(lines) == 1:
        value = [float(number) for number in lines[0].split(',')]
    else:
        value = [[float(number) for number in lines[row].split(',')]
                 for row in range(len(lines))]

    return value


def load_dataset(vocab_size):
    """Loads the train and test set."""
    train_data_reader = DataReader(
        data_path='./datasets/20news_train_tfidf.txt',
        batch_size=128,
        vocab_size=vocab_size
    )
    test_data_reader = DataReader(
        data_path='./datasets/20news_test_tfidf.txt',
        batch_size=128,
        vocab_size=vocab_size
    )
    return train_data_reader, test_data_reader


def model(train_data_reader, test_data_reader, vocab_size, hidden_size=50, learning_rate=0.0001, num_epochs=15):
    m = train_data_reader._data.shape[0]            # number of training examples
    batch_size = train_data_reader._batch_size      # batch size
    seed = 3
    costs = []                                      # keep track of the cost

    # create a computational graph
    mlp = MLP(vocab_size=vocab_size, hidden_size=hidden_size)
    predicted_labels, loss = mlp.build_graph()
    optimizer = mlp.trainer(loss, learning_rate=learning_rate)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        num_mini_batches = int(m / batch_size)   # number of minibatches of size minibatch_size in the train set

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            seed = seed + 1  # Increases the seed value
            mini_batches = train_data_reader.random_mini_batches(seed)

            for mini_batch in mini_batches:
                # Select a mini batch
                (mini_batch_data, mini_batch_labels) = mini_batch

                # Run the session to execute the "optimizer" and the "loss"
                _, minibatch_cost =\
                    sess.run([optimizer, loss],
                            feed_dict={mlp._X: mini_batch_data, mlp._real_Y: mini_batch_labels})

                epoch_cost += minibatch_cost / num_mini_batches

            # print the cost
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # save parameters
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            save_parameters(
                name=variable.name,
                value=variable.eval(),
                epoch=num_epochs
            )

        # compute the predicted labels
        predicted_labels_train = sess.run([predicted_labels],
                                feed_dict={
                                    mlp._X: train_data_reader._data,
                                    mlp._real_Y: train_data_reader._labels
                                })
        predicted_labels_test = sess.run([predicted_labels],
                                          feed_dict={
                                              mlp._X: test_data_reader._data,
                                              mlp._real_Y: test_data_reader._labels
                                          })

        # change the type of labels from list to numpy array
        predicted_labels_train = np.array(predicted_labels_train)
        predicted_labels_test = np.array(predicted_labels_test)

        # compute the accuracy
        train_acc = np.mean(np.equal(predicted_labels_train.T, train_data_reader._labels))
        test_acc = np.mean(np.equal(predicted_labels_test.T, test_data_reader._labels))

        # print the accuracy
        print("Train accuracy: ", train_acc)
        print("Test accuracy: ", test_acc)


if __name__=="__main__":
    # get the vocab_size
    with open('./datasets/words_idfs.txt') as f:
        vocab_size = len(f.readlines())

    # load data set
    train_data_reader, test_data_reader = load_dataset(vocab_size)

    # run the model and save the parameters
    model(train_data_reader,test_data_reader, vocab_size=vocab_size, hidden_size=50, learning_rate=0.001, num_epochs=15)
    # Train accuracy: 0.9992929114371575
    # Test accuracy: 0.8437334041423261
