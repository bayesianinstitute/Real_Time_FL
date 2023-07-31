import json
import torch.optim as optim
import os
import torch
import socket
import pickle

from web3 import Web3, HTTPProvider
from BCCommunicator import BCCommunicator
from FSCommunicator import FSCommunicator

import ipfshttpclient
import io

import torch
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import random
import time
from Worker_Main import Worker

if __name__ == '__main__':
    ipfs_path = 'QmdzVYP8EqpK8CvH7aEAxxms2nCRNc98fTFL2cSiiRbHxn'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_evil = False
    topk = 1
    HOST = 'localhost'
    PORT = 12346
    client_port = random.randint(40000, 50000)
    client_port_next = random.randint(50000, 60000)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    worker_dict = OrderedDict()

    # Reuse the socket address to avoid conflicts when restarting the program
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the worker's socket to the specified port
    client_socket.bind(('localhost', client_port))  # Bind to all available network interfaces

    client_socket.connect((HOST, PORT))
    print("Connected to server")
    current_port = client_socket.getsockname()[1]
    print("current port : ", current_port)
    worker = Worker(ipfs_path, device, is_evil, topk)
    worker.send_data(client_socket, client_port_next)

    received_json = worker.receive_data(client_socket)
    print("received_json : ", received_json)
    received_headid = worker.receive_data(client_socket)
    print("received_headid : ", received_headid)

    while True:
        contract_address = '0xdD0751275E7e9fE7c35798Ca124F970F5755Fb26'
        workerAddress = worker.workerAddress()

        client_socket.close()
        print("Connection close from Application")

        print("Training Model")
        print("received_headid : ", received_headid)

        weights = worker.train(round=1)

        is_header = True
        worker_dict = OrderedDict()
        if received_headid['new_port'] == client_port_next:
            print("I am the header")

            server_socket_peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket_peer.bind(('localhost', client_port_next))  # Bind to all available network interfaces
            server_socket_peer.listen(2)

            client_sockets = []

            for i in range(2):
                client_socket, addr = server_socket_peer.accept()
                print("Connection from:", addr)
                client_sockets.append(client_socket)

            print("Connected to peer")

            worker_weights = []
            for idx, client_socket in enumerate(client_sockets):
                work_address = worker.receive_data(client_socket)
                print("Receive data from client", idx + 1)
                worker_weights.append(work_address)

            # Assuming you want to store the worker addresses in the worker_dict
            for idx, weight in enumerate(worker_weights):
                # The key will be in the format 'worker_1_weights', 'worker_2_weights', and so on
                key = f'worker_{idx + 1}_weights'
                # Add the weight to the OrderedDict with the corresponding key
                worker_dict[key] = weight

            averaged_weights = worker.average(worker_dict)
            print("Averaged weights are Done")

            try:
                for idx, client_socket in enumerate(client_sockets):
                    print("Sending weight to client:", idx + 1)
                    worker.send_data(client_socket, averaged_weights)
                    print("Sent new weights to clients", idx + 1)

            except ConnectionResetError:
                # Handle the case when a client disconnects unexpectedly
                print("Client", idx + 1, "disconnected.")
                client_sockets.pop(idx)

            worker.update_model(averaged_weights)
            print("Worker Update it works")

            file_name = 'worker_data.json'
            worker_head_id = worker.shuffle_worker_head(received_json)
            print("suffle_id id ", worker_head_id)
            print("client_port_next_id ", client_port_next)

            old_client_port_next = client_port_next

            # worker_head_id = 2  # Example worker ID for demonstration

            if worker_head_id != client_port_next:
                client_port_next = random.randint(50000, 60000)
                # Find the dictionary with 'workerid' equal to worker_head_id and update its 'new_port' value
                for entry in received_json:
                    if entry['workerid'] == 2:
                        entry['new_port'] = client_port_next
                        break

                # Write the updated JSON data back to the file
            
            received_headid=worker_head_id

            try:
                for idx, client_socket in enumerate(client_sockets):
                    print("Sending json file to client:", idx + 1)
                    worker.send_data(client_socket, worker_head_id)
                    worker.send_data(client_socket, received_json)
                # Receive acknowledgment from each client after sending the data

            except ConnectionResetError:
                # Handle the case when a client disconnects unexpectedly
                print("Client", idx + 1, "disconnected.")
                client_sockets.pop(idx)

            except Exception as e:
                print("Error sending data", e)

            print("old port {} and new port {}".format(old_client_port_next, client_port_next))

            client_sockets = []

            # Check if the worker's ID matches the new shuffled ID
            if received_headid['new_port'] != old_client_port_next:
                # If the worker is no longer the header, exit the header loop
                print("I am no longer the header.")
                is_header = False
            else:
                print("I am the header Again.")
                is_header = True
        else:
            peer_ip = received_headid['address']
            peer_port = received_headid['new_port']

            print("peer ip {} and port {}  ".format(peer_ip, peer_port))

            try:
                client_socket_peer = worker.connect_to_peer(peer_ip, peer_port)
                worker.send_data(client_socket_peer, weights)
                print("Worker Sending Weights to peer")

                print("received_json",received_json)


                average_Weight = worker.receive_data(client_socket_peer)
                print("Got Average Weight")
                worker.update_model(average_Weight)
                print("Updated model weights")

                received_headid = worker.receive_data(client_socket_peer)
                print("received_headid : ", received_headid)
                received_json = worker.receive_data(client_socket_peer)
                print("new received_json : ", received_json)

                if received_headid['new_port'] == client_port_next:
                    print("I am the header again.")
                    is_header = True
                else :
                    is_header =False
                    
            except Exception as e:
                print("Error during peer connection:", e)

