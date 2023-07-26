import socket
import json

# Server socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("localhost", 12347))
server_socket.listen(2)

clients = []  # Store client information (IP, port)

while len(clients) < 2:
    print("Waiting for clients to connect...")
    client_socket, client_address = server_socket.accept()
    print(f"Connected to {client_address}")
    clients.append(client_address)

# Send client information to each client
clients_data = json.dumps(clients).encode()
for client_socket, client_address in zip(clients, clients):
    client_socket.send(clients_data)

server_socket.close()
