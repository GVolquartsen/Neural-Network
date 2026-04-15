from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import threading
import json
import network

app = Flask(__name__)

training_thread = None
is_training = False

# Serve the main page
@app.route('/')
def index():
    return render_template('index.html')


# Stream training progress using Server Sent Events
@app.route('/train', methods=['POST'])
def train():
    global is_training

    # Get settings from the request
    data = request.get_json()
    epochs = int(data.get('epochs', 75))
    batch_size = int(data.get('batch_size', 64))
    network.initial_alpha = float(data.get('alpha', 0.1))
    network.decay_rate = float(data.get('decay_rate', 0.99))
    network.dropout_rate = float(data.get('dropout_rate', 0.2))

    # Handle architecture changes — reinitialize weights if needed
    new_n = data.get('n', None)
    new_L = data.get('L', None)
    if new_n and new_L:
        network.n = new_n
        network.L = new_L
        network.W = [None]
        network.b = [None]
        for l in range(1, network.L + 1):
            network.W.append(np.random.randn(network.n[l], network.n[l-1]) * np.sqrt(2 / network.n[l-1]))
            network.b.append(np.zeros((network.n[l], 1)))

    def generate():
        global is_training
        is_training = True
        num_batches = network.m // batch_size

        for epoch in range(epochs):
            if network.stop_training:
                network.stop_training = False
                yield f"data: {json.dumps({'stopped': True})}\n\n"
                return
            # Shuffle data
            shuffle = np.random.permutation(network.m)
            network.A0 = network.A0[:, shuffle]
            network.Y = network.Y[:, shuffle]
            network.alpha = network.initial_alpha * network.decay_rate ** epoch

            for i in range(num_batches):
                a = network.A0[:, i * batch_size: (i + 1) * batch_size]
                y = network.Y[:, i * batch_size: (i + 1) * batch_size]
                A, Z = network.feed_forward(a)
                network.backprop(A, Z, y, batch_size)

            # Compute stats
            A_full, _ = network.feed_forward(network.A0)
            full_cost = network.cost(A_full[network.L], network.Y)
            train_acc = network.accuracy(network.A0, network.Y)
            test_acc = network.accuracy(network.testX, network.testY)

            # Stream stats to browser as SSE
            payload = json.dumps({
                'epoch': epoch + 1,
                'epochs': epochs,
                'cost': round(full_cost, 4),
                'train_acc': round(train_acc * 100, 2),
                'test_acc': round(test_acc * 100, 2)
            })
            yield f"data: {payload}\n\n"

        # Save weights and signal completion
        network.save_weights()
        yield f"data: {json.dumps({'done': True})}\n\n"
        is_training = False

    return Response(generate(), mimetype='text/event-stream')


# Handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = np.array(data['pixels']).reshape(784, 1) / 255.0

    pred = network.make_prediction(pixels)[0]
    A_out, _ = network.feed_forward(pixels, training=False)
    probs = A_out[network.L].flatten().tolist()
    confidence = round(max(probs) * 100, 1)

    return jsonify({
        'prediction': int(pred),
        'confidence': confidence,
        'probabilities': [round(p * 100, 1) for p in probs]
    })


# Load saved weights
@app.route('/load_weights', methods=['POST'])
def load_weights_route():
    try:
        network.load_weights()
        test_acc = network.accuracy(network.testX, network.testY)
        return jsonify({'success': True, 'test_acc': round(test_acc * 100, 2)})
    except FileNotFoundError:
        return jsonify({'success': False, 'error': 'No saved weights found'})


if __name__ == '__main__':
    # Load data before starting server
    print("Loading MNIST data...")
    network.A0, network.Y, network.m, network.testX, network.testY = network.prepare_data()
    print("Data loaded! Starting server...")
    app.run(debug=False)