import numpy as np
import os
import pickle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# 1 - Create Network Architecture
L = 4  # 3 hidden layers 1 output layer
n = [784, 256, 128, 64, 10]
initial_alpha = 0.1
decay_rate = 0.99
alpha = initial_alpha
dropout_rate = 0.20
stop_training = False
m = 0  # will be set after prepare_data() is called
A0 = None
Y = None
testX = None
testY = None

# 2 - Generate weights and biases
W = [None]
b = [None]
for l in range(1, L + 1):
    W.append(np.random.randn(n[l], n[l-1]) * np.sqrt(2 / n[l-1]))
    b.append(np.zeros((n[l], 1)))


def prepare_data():
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test_raw, y_test_raw) = mnist.load_data()

    m = y_train.shape[0]
    m_test = y_test_raw.shape[0]
    labels = y_train

    # Initialize label matrix Y
    Y = np.zeros((10, m))
    Y[labels, np.arange(m)] = 1
    testLabels = y_test_raw

    # Flattening the inputs. 1 picture per column, and then normalizing it
    X = X_train.reshape(m, 784).T / 255
    X_test = X_test_raw.reshape(m_test, 784).T / 255
    Y_test = np.zeros((10, m_test))
    Y_test[testLabels, np.arange(m_test)] = 1

    return X, Y, m, X_test, Y_test


# 4 - Activation functions
def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x <= 0, 0, 1)


def softmax(x):
    return np.exp(x - np.max(x, axis=0, keepdims=True)) / np.sum(
        np.exp(x - np.max(x, axis=0, keepdims=True)), axis=0, keepdims=True)


# 5 - Feed Forward Process
def feed_forward(A0, training=True):
    A = [A0]
    Z = [None]

    for l in range(1, L + 1):
        # layer l calculations
        Z_l = W[l] @ A[l - 1] + b[l]
        # if output layer, use softmax, if hidden layer, use relu
        if l == L:
            A_l = softmax(Z_l)
        else:
            A_l = relu(Z_l)
            # apply dropout to hidden layers only during training
            if training:
                mask = np.random.rand(*A_l.shape) > dropout_rate
                A_l = A_l * mask / (1 - dropout_rate)
        Z.append(Z_l)
        A.append(A_l)

    return A, Z


def cost(y_hat, y):
    m_local = y.shape[1]
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return -(1 / m_local) * np.sum(y * np.log(y_hat))


def backprop(A, Z, Y, m):
    global W, b
    dZ = (A[L] - Y)

    for l in range(L, 0, -1):
        dW = (1 / m) * (dZ @ A[l - 1].T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = W[l].T @ dZ

        if l > 1:
            dZ = dA * relu_derivative(Z[l - 1])

        W[l] = W[l] - dW * alpha
        b[l] = b[l] - db * alpha


# Saving weights
def save_weights():
    with open('weights.pkl', 'wb') as f:
        pickle.dump((W, b), f)

# Loading weights
def load_weights():
    global W, b
    with open('weights.pkl', 'rb') as f:
        W, b = pickle.load(f)
def make_prediction(x):
    A, Z = feed_forward(x, training=False)
    y = np.argmax(A[L], axis=0)
    return y


def accuracy(X, Y):
    predictions = make_prediction(X)
    actual = np.argmax(Y, axis=0)
    return np.mean(predictions == actual)


def training_loop():
    global A0, Y, alpha
    epochs = 10  # Amount of passes through data set 60,000 * epochs
    batch_size = 64
    num_batches = m // batch_size

    for epoch in range(epochs):
        # Generate a random ordering of indices
        shuffle = np.random.permutation(m)

        # Use that ordering to reorder both arrays consistently
        A0 = A0[:, shuffle]
        Y = Y[:, shuffle]
        alpha = initial_alpha * decay_rate ** epoch

        for i in range(num_batches):
            # Slices A0 and Y to get the current batch
            a = A0[:, i * batch_size: (i + 1) * batch_size]
            y = Y[:, i * batch_size: (i + 1) * batch_size]
            A, Z = feed_forward(a)
            c = cost(A[L], y)
            backprop(A, Z, y, batch_size)

        A_full, Z_full = feed_forward(A0)
        full_cost = cost(A_full[L], Y)
        print(f"Epoch {epoch}, Full Cost: {full_cost}, Accuracy: {accuracy(A0, Y)}")


if __name__ == "__main__":
    A0, Y, m, testX, testY = prepare_data()

    # Get User choice
    choice = input("1 for Saved Weights, 2 to start fresh: ")
    if choice == "1":
        load_weights()
    elif choice == "2":
        training_loop()
        save_weights()

    # Always runs regardless of choice
    image = A0[:, 0:1]
    prediction = make_prediction(image)
    print(f"Predicted: {prediction}")
    print(f"Actual Value: {np.argmax(Y[:, 0])}")
    print(f"Test accuracy: {accuracy(testX, testY)}")