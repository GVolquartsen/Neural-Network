# MNIST Digit Recognizer

A fully connected neural network built from scratch using only NumPy, trained on the MNIST handwritten digit dataset to recognize handwritten digits 0-9. The network implements every component manually — forward propagation, backpropagation, mini-batch gradient descent, He initialization, ReLU activations, Softmax output, categorical cross entropy loss, learning rate decay, and inverted dropout regularization — with no deep learning frameworks involved. The architecture is fully configurable, allowing you to adjust the number of layers, neurons per layer, learning rate, dropout rate, and training epochs. A Flask web interface lets you draw digits with your mouse and see live predictions alongside a probability bar chart for all 10 digits, while a real-time training dashboard tracks cost and accuracy epoch by epoch. Trained weights can be saved and reloaded between sessions so you don't have to retrain from scratch every time.

## Getting Started
```
pip install numpy flask tensorflow
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.
