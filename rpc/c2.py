import socket
import json

# Connect to the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.connect(("localhost", 12347))
data = server_socket.recv(1024).decode()
clients = json.loads(data)

# Extract the IP and port of Client 1
client1_ip, client1_port = clients[0]

# Client 2 sets up a socket to communicate directly with Client 1
client2_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client2_socket.connect((client1_ip, client1_port))

while True:
    message = input("Client 2: ")
    client2_socket.send(message.encode())
    # Add any necessary code for receiving data from Client 1 here
