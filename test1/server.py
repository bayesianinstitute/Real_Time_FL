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
    file_path='server_model.h5'
    model = tf.keras.models.load_model(file_path)
    # model.fit(X_train, y_train, epochs=2, batch_size=200)

    return model

def send_data(socket, data):
    try:
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        socket.sendall(data_size.to_bytes(4, 'big'))  # Sending the size of data first
        socket.sendall(serialized_data)              # Sending the actual data
    except BrokenPipeError:
        print("Client disconnected during send_data.")
        raise            # Sending the actual data

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

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(2)  # Allow two client connections
    print("Waiting for connections from clients...")

    # Accept connections from two clients
    client_sockets = []
    for _ in range(2):
        client_socket, addr = server_socket.accept()
        print("Connection from:", addr)
        client_sockets.append(client_socket)

    # Load MNIST data
    (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape((60000, 784)).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Build the model and train
    model = build_model()
    model = train_model(model, X_train, y_train)

    # Iterative collaborative training
    for _ in range(5):  # 5 rounds of collaborative training
        client_weights = []

        # Send the current model weights to each client
        for client_socket in client_sockets:
            weights = model.get_weights()
            send_data(client_socket, weights)

        # Receive updated weights from each client and aggregate them
        for client_socket in client_sockets:
            updated_weights = receive_data(client_socket)
            client_weights.append(updated_weights)

        # Aggregate the received weights
        averaged_weights = [sum(w) / len(w) for w in zip(*client_weights)]
        model.set_weights(averaged_weights)

        # Send the updated model weights back to each client
        for client_socket in client_sockets:
            send_data(client_socket, averaged_weights)

    # Save the final model
    model.save('server_modeled.h5')

    # Close client connections
    for client_socket in client_sockets:
        client_socket.close()

if __name__ == "__main__":
    main()