import socket

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 12345))

    while True:
        message = input("Enter a message to send to the server (or 'exit' to quit): ")
        if message.lower() == "exit":
            break

        client_socket.sendall(message.encode())
        data = client_socket.recv(1024)
        print(f"Received from server: {data.decode()}")

    print("Connection closed.")
    client_socket.close()

if __name__ == "__main__":
    start_client()
