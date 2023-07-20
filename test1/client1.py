import socket
import pickle
import numpy as np
import tensorflow as tf

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=1, batch_size=200)  # Local training for 1 epoch
    return model

def send_data(socket, data):
    serialized_data = pickle.dumps(data)
    data_size = len(serialized_data)
    socket.sendall(data_size.to_bytes(4, 'big'))  # Sending the size of data first
    socket.sendall(serialized_data)              # Sending the actual data

def receive_data(socket):
    data_size_bytes = socket.recv(4)
    data_size = int.from_bytes(data_size_bytes, 'big')
    serialized_data = b''
    while len(serialized_data) < data_size:
        remaining = data_size - len(serialized_data)
        serialized_data += socket.recv(4096 if remaining > 4096 else remaining)
    return pickle.loads(serialized_data)

def main():
    host = '127.0.0.1'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Load MNIST data
    _, (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 784)).astype('float32') / 255
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build the model
    model = build_model()

    for _ in range(5):  # 5 rounds of collaborative training
        # Receive the current model weights from the server
        weights = receive_data(client_socket)
        model.set_weights(weights)

        # Train the model on client-side with local data
        model = train_model(model, X_test, y_test)

        # Send the updated weights back to the server
        updated_weights = model.get_weights()
        send_data(client_socket, updated_weights)

    # client_socket.close()

if __name__ == "__main__":
    main()
