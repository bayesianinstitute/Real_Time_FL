import socket

def create_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def bind_socket(sock, port):
    sock.bind(('localhost', port))

def connect_socket(sock, port):
    sock.connect(('localhost', port))

def send_data(sock, data):
    sock.send(data.encode())

def receive_data(sock):
    data, _ = sock.recvfrom(1024)
    return data.decode()
