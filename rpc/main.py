import socket
import threading

# Server configurations
SERVER_ADDRESSES = [("127.0.0.1", 8000), ("127.0.0.1", 8001), ("127.0.0.1", 8002)]  # Add your server IPs and ports here

# Load balancing variables
current_server_index = 0
lock = threading.Lock()

def handle_client(client_socket, target_socket):
    while True:
        data = client_socket.recv(4096)
        if not data:
            break
        # Forward data from the client to the server
        target_socket.send(data)
        response = target_socket.recv(4096)
        if not response:
            break
        # Send the server response back to the client
        client_socket.send(response)
    client_socket.close()
    target_socket.close()

def main():
    global current_server_index

    # Create a load balancer socket
    balancer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    balancer_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    balancer_socket.bind(("0.0.0.0", 8888))  # Load balancer listens on port 8888
    balancer_socket.listen(5)

    print("Load balancer is running on port 8888...")

    while True:
        client_socket, client_address = balancer_socket.accept()
        print("Accepted connection from:", client_address)

        # Select the server to forward the client to (Round Robin)
        with lock:
            current_server_index = (current_server_index + 1) % len(SERVER_ADDRESSES)
            target_server_address = SERVER_ADDRESSES[current_server_index]

        # Connect to the selected server
        target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_socket.connect(target_server_address)

        # Start handling the client and server communication in separate threads
        client_thread = threading.Thread(target=handle_client, args=(client_socket, target_socket))
        client_thread.start()

if __name__ == "__main__":
    main()