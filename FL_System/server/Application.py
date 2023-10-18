# from futuretest.BCCommunicator import BCCommunicator
# from futuretest.FSCommunicator import FSCommunicator
# from futuretest.Model import Model
# from futuretest.BCCommunicator import BCCommunicator
import torch
import os
from Requester import Requester
from dotenv import load_dotenv
# from futuretest.FSCommunicator import FSCommunicator
# import ipfshttpclient
import socket
import pickle
import json 
import random
from collections import OrderedDict
from config_app import HOST,PORT
import sys

# Main class to simulate the distributed application
class Application:

    def __init__(self, num_workers, num_rounds, num_evil=0):
        # self.client = ipfshttpclient.connect()
    
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.fspath ='QmdzVYP8EqpK8CvH7aEAxxms2nCRNc98fTFL2cSiiRbHxn'
        self.workers = []
        self.topk = num_workers
        self.worker_dict =  OrderedDict()
        self.num_evil = num_evil
        self.ffi = None  # Add the FFI attribute and set it to None
        self.contract_address = None
        self.worker_address = {}

        
 
    def send_data(self,socket, data):
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        socket.sendall(data_size.to_bytes(4, 'big'))  # Sending the size of data first
        socket.sendall(serialized_data)              # Sending the actual data   


    def receive_data(self,socket):
        # Receiving the size of data first
        data_size_bytes = socket.recv(4)
        data_size = int.from_bytes(data_size_bytes, 'big')

        # Receiving the serialized data
        serialized_data = b""
        while len(serialized_data) < data_size:
            remaining_bytes = data_size - len(serialized_data)
            serialized_data += socket.recv(remaining_bytes)

        # Decoding the serialized data using pickle
        data = pickle.loads(serialized_data)
        return data
    


    def average(self, worker_weights):
        all_keys_list = list(worker_weights.keys())

        print("all keys: ", all_keys_list)

        averaged_weights = OrderedDict()
        for layer_key in worker_weights[all_keys_list[0]]:
            layer_weights = [worker_weights[worker][layer_key] for worker in worker_weights]
            averaged_weights[layer_key] = torch.stack(layer_weights).mean(dim=0)


        return averaged_weights

    def save_worker_data_to_json(self, worker_data):
        # Save the worker data dictionary to JSON file
        worker_data_1 = []
        worker_data_2 = []

        for data in worker_data:
            if data['location'] == 'INDIA':
                worker_data_1.append(data)
            elif data['location'] == 'USA':
                worker_data_2.append(data)

        # Save Cluster 1 data to JSON file
        with open('C1_worker_data.json', 'w') as file:
            json.dump(worker_data_1, file)

        # Save Cluster 2 data to JSON file
        with open('C2_worker_data.json', 'w') as file:
            json.dump(worker_data_2, file)

    def load_worker_data_from_json(self,files):
        # Load the worker data from JSON file
        with open(files, 'r') as file:
            worker_data = json.load(file)
        return worker_data
    
    def shuffle_worker_head(self,worker_head_ids):
    # Randomly shuffle the worker head IDs to select one for each round
        random.shuffle(worker_head_ids)
        return worker_head_ids[0]
    
    def load_worker_head_ids(self,json_file):
    # Load the JSON file and extract worker head IDs
        with open(json_file, 'r') as f:
            worker_head_ids = json.load(f)
        return worker_head_ids


    def choose_random_worker(self):
        # Load worker data from the JSON file
        worker_data = self.load_worker_data_from_json()
        # Choose a random worker ID
        random_worker_id = random.choice(list(worker_data.keys()))
        return worker_data[random_worker_id]

    def send_worker_data_to_clients(self, worker_data):
        # Serialize the worker data
        serialized_worker_data = json.dumps(worker_data).encode()

        # Send the worker data to all clients
        for client_socket in client_socket:
            client_socket.send(serialized_worker_data)

    
    def run(self):
        load_dotenv()
        print("Task initialized")

        # Create a socket

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(3)  # Allow Six client connections
        print("Waiting for connections from clients...")

        # Accept connections from two clients
        # Create a list to store client sockets and worker data
        client_sockets = []
        worker_info_list = []
        # Accept connections from worker clients and store their socket information
        for i in range(3):
            client_socket, addr = server_socket.accept()
            print("Connection from:", addr)
            client_sockets.append(client_socket)

            # You can also add any additional information about the worker here if needed
            client_info = {
                'address': addr[0],
                'port': addr[1],
                'workerid': i+1,
                
            }
            worker_info_list.append(client_info)

        print("Received all client connections")
        new_port=[]        
        # Receive Meta Data
        for idx, client_socket in enumerate(client_sockets):
            meta = self.receive_data(client_socket)
            print(f"{idx+1} meta data : {meta}" )


            port=meta['port'] 
            locations=meta['locations'] 
            cluster_head=meta['cluster_head_port']
            # print("Port", port)
            # print("Location", locations)
            new_port.append(port)
            # Update the 'port,location and cluster_head_port' value in the client_info dictionary
            worker_info_list[idx]['new_port'] = port
            worker_info_list[idx]['location'] = locations
            worker_info_list[idx]['cluster_head_port']=cluster_head
        
        # Save the updated worker information with new ports to the JSON file
        self.save_worker_data_to_json(worker_info_list)
        print("Worker information with new ports saved")
        while True :
            file_name_1='C1_worker_data.json'
            # file_name_2='C2_worker_data.json'

            file_json_1=self.load_worker_data_from_json(file_name_1)
            worker_head_ids_1 = self.load_worker_head_ids(file_name_1)
            worker_head_id_1 = self.shuffle_worker_head(worker_head_ids_1)

            print("suffle_id id ",worker_head_id_1)
            try:
                for idx,client_socket in enumerate(client_sockets[:3]):
                    print("Sending json file to client for Cluster 1:",idx+1)
                    import time
                    time.sleep(4)
                    
                    self.send_data(client_socket, file_json_1)
                    self.send_data(client_socket,worker_head_id_1)
                    # self.send_data(client_socket,worker_head_id_2)
            except ConnectionResetError:
                # Handle the case when a client disconnects unexpectedly
                print("Client", idx + 1, "disconnected.")
                client_sockets.pop(idx)

            except Exception as e:
                print("Error sending data",e)

            print("Send file to all clusters")

            print("Done from Application")

            # Exit the program
            sys.exit()

        