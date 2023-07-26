import socket
import json

# Connect to the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.connect(("localhost", 12347))
data = server_socket.recv(1024).decode()
clients = json.loads(data)

# Extract the IP and port of Client 2
client2_ip, client2_port = clients[1]

# Client 1 sets up a socket to communicate directly with Client 2
client1_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client1_socket.connect((client2_ip, client2_port))

while True:
    message = input("Client 1: ")
    client1_socket.send(message.encode())
    # Add any necessary code for receiving data from Client 2 here
