from socket1 import create_socket as create_socket1, bind_socket as bind_socket1, connect_socket as connect_socket1, send_data as send_data1, receive_data as receive_data1
from socket2 import create_socket as create_socket2, bind_socket as bind_socket2, connect_socket as connect_socket2, send_data as send_data2, receive_data as receive_data2

# Socket 1 usage
socket1 = create_socket1()
bind_socket1(socket1, 1000)
connect_socket1(socket1, 2000)

# Socket 2 usage
socket2 = create_socket2()
bind_socket2(socket2, 2001)
connect_socket2(socket2, 1000)

# Perform operations with Socket 1 and Socket 2
send_data1(socket1, "Hello from Socket 1!")
received_data = receive_data2(socket2)
print("Received from Socket 2:", received_data)

send_data2(socket2, "Hi from Socket 2!")
received_data = receive_data1(socket1)
print("Received from Socket 1:", received_data)

# Close the sockets
socket1.close()
socket2.close()
