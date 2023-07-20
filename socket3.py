import socket

def receive_data(server_socket):
    return server_socket.recv(1024).decode('utf-8')

def send_data(server_socket, data):
    server_socket.send(data.encode('utf-8'))

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server's IP address and port
server_ip = 'localhost'
port = 12345
client_socket.connect((server_ip, port))

while True:
    message = input("Enter a message to send to the server (or type 'exit' to quit): ")
    if message.lower() == 'exit':
        break

    send_data(client_socket, message)

    response = receive_data(client_socket)
    print("Server response:", response)

# Close the connection
client_socket.close()
