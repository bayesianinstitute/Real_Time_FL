import socket

def receive_data(client_socket):
    return client_socket.recv(1024).decode('utf-8')

def send_data(client_socket, data):
    client_socket.send(data.encode('utf-8'))

def handle_client(client_socket, client_id):
    while True:
        data_received = receive_data(client_socket)
        if not data_received:
            break

        print(f"Received from Client {client_id}: {data_received}")

        response = input(f"Enter a response to send to Client {client_id} (or type 'exit' to quit): ")
        if response.lower() == 'exit':
            break

        send_data(client_socket, response)

    client_socket.close()

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific IP address and port
host = 'localhost'
port = 12345
server_socket.bind((host, port))

# Enable the server to accept connections (3 connections at a time)
server_socket.listen(3)

print("Waiting for clients to connect...")

# Accept connections from three clients
clients = []
for i in range(2):
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with Client {i+1}: {client_address}")
    clients.append(client_socket)

# Handling communication for all three clients
for i, client_socket in enumerate(clients):
    handle_client(client_socket, i+1)

# Close all client connections
for client_socket in clients:
    client_socket.close()

# Close the server socket
server_socket.close()
