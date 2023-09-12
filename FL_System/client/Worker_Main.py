import json
import torch.optim as optim
import os
import torch
import socket
import pickle

from web3 import Web3, HTTPProvider


import ipfshttpclient
import io

import torch
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import random
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from dotenv import load_dotenv
from  threading import Thread 



class Model():
    '''
    Contains all machine learning functionality
    '''
    # static, might have to be calculated dynamically
    batch_size = 64
    epochs = 1

    def __init__(self,  model, optimizer, device, topk, isEvil = False):
        self.num_workers = 3
        self.idx = 1
        self.model = model
        self.optimizer = optimizer
        self.DEVICE = device
        self.topk = topk
        self.isEvil = isEvil
        
        
        # this would be generic in a real application
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size, shuffle=True)
        
        self.garbage = torch.rand((64,1,28,28))
        
        # find the datasets indices
        # also this would not be implemented like this in the real application
        # the users would use an 80/20 random split of their own dataset for training/validating
        self.num_train_batches = len(self.train_loader)//self.num_workers
        self.num_test_batches = len(self.test_loader)//self.num_workers 
        # start idx
        self.start_idx_train = self.num_test_batches* self.idx
        self.start_idx_test = self.num_test_batches * self.idx
        
        
    def average(self, state_dicts):
        # terrible way to do it
        final_dict = {}
        for key in state_dicts[0]:
            final_dict[key] = torch.clone(state_dicts[0][key])
            for i in state_dicts[1:]:
                final_dict[key] += torch.clone(i[key])
            final_dict[key]/=self.num_workers

        return final_dict
    
    
    def adapt_current_model(self, avg_state_dict):
        self.model.load_state_dict(avg_state_dict)


    def train(self):
        
        self.model.train()
        
        for epoch in range(self.epochs):
            for idx, (data, target) in enumerate(self.train_loader):
                if idx >= self.start_idx_train and idx < self.start_idx_train + self.num_train_batches:
                    self.optimizer.zero_grad()
                    if not self.isEvil:
                        output = self.model(data.to(self.DEVICE))
                    else:
                        output = self.model(self.garbage.to(self.DEVICE))
                        target = torch.randint(0, 8, (64,1)).reshape(64)
                    loss = F.nll_loss(output, target.to(self.DEVICE))
                    loss.backward()
                    self.optimizer.step()
            print('finished epoch {} of worker {}'.format(epoch, self.idx))
        return self.model.state_dict()

    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                if idx >= self.start_idx_test and idx < self.start_idx_test + self.num_test_batches:
                    output = self.model(data.to(self.DEVICE))
                    test_loss += F.nll_loss(output.to(self.DEVICE), target.to(self.DEVICE)).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.to(self.DEVICE).view_as(pred)).sum()

        test_loss /= (self.num_test_batches * self.batch_size)
        accuracy = 100. * correct / (self.num_test_batches * self.batch_size)

        print('\nTest set: Accuracy: {}/{} ({:.0f}%), Loss: {:.6f}\n'.format(
            correct, self.num_test_batches * self.batch_size, accuracy, test_loss))
        
        return accuracy, test_loss
    
    def eval(self, model_state_dicts,id):
        res = []
        for idx, m in enumerate(model_state_dicts):
            pass
            # self.model.load_state_dict(m)
        print("idx : ", id)
        acc = self.test()
        res.append((acc,id,m))
            
        print('length {}  of  res :{} '.format(len(res),res))
        sorted_models = sorted(res, key=lambda t: t[0])
        # return self.rank_models(sorted_models),  self.get_top_k(sorted_models), res
        return res
            
            
            

class Worker(Thread):
    truffle_file = json.load(open('./build/contracts/FLTask.json'))

    

    def __init__(self,  device, is_evil, topk,worker_id,keyy):
        # self.bcc = BCCommunicator()
        # self.fsc = FSCommunicator(ipfs_path, device)
        load_dotenv()


        #  ipfs connection and blockchain key
        # self.key = str(input("Enter the private key"))
        self.key = keyy
        
        # self.client_url = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')


        # self.model_hash = 'QmZaeFLUPJZopTvKWsuji2Q5RPWaTLBRtpCoBbDM6sDyqM'
        # model_bytes = self.client_url.cat(self.model_hash)
        # model = torch.jit.load(io.BytesIO(model_bytes),
        #                        map_location=device)
        model= torch.load("fs-sim/model.pt")
        print("Done Model!!!!!!")

        # optimizer_hash = 'Qmd96G9irL6hQuSfGFNoYqeVgg8DvAyv6GCt9CqCsEDj1w'
        # optimizer_bytes = self.client_url.cat(optimizer_hash)
        # opt = torch.load(io.BytesIO(
        #     optimizer_bytes), map_location=device)
        opt=torch.load("fs-sim/optimizer.pt")
        print("Done Optimizer !!!!!")



        self.is_evil = is_evil

        class_ = getattr(optim, opt['name'])
        copy = dict(opt['state_dict']['param_groups'][0])
        try:
            del copy['params']
        except:
            pass
        opt = class_(model.parameters(), **(copy))

        self.model = Model(model, opt, device, topk)

        self.num_workers = 0
        print("key", self.key)

        key = self.key

        self.PROJECT_API=os.getenv('PROJECT_API')
        print("PROJECT_API :",self.PROJECT_API)
        # init web3.py instance

        # blockchain Connection
        self.w3 = Web3(HTTPProvider(self.PROJECT_API))


        if self.w3.isConnected():
            print("Worker initialization: connected to blockchain")

        self.account = self.w3.eth.account.privateKeyToAccount(key)
        self.contract = self.w3.eth.contract(bytecode=self.truffle_file['bytecode'], abi=self.truffle_file['abi'])

    def join_task(self, contract_address):
        self.contract_address = contract_address
        self.contract_instance = self.w3.eth.contract(abi=self.truffle_file['abi'], address=contract_address)
        deposit = 100000000000000000  # 0.1 ethers (in Wei)
        tx = self.contract_instance.functions.joinTask().buildTransaction({
            "gasPrice": self.w3.eth.gas_price,
            "chainId": 1337,
            "from": self.account.address,
            "value": deposit,
            'nonce': self.w3.eth.getTransactionCount(self.account.address)
        })
        # Get tx receipt to get contract address
        signed_tx = self.w3.eth.account.signTransaction(tx, self.key)
        tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.getTransactionReceipt(tx_hash)

        
 


    def workerAddress(self):
        print("Addressing: ", self.account.address)
        return self.account.address

    def train(self, round):
        # Train the model
        cur_state_dict = self.model.train()
        print("Modle Summary",self.model)
        # print("Training state: ", cur_state_dict)
        return cur_state_dict
    
    def test(self):

        return self.model.test()


    def send_data(self,socket, data):
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        print("data size: ", data_size)
        try: 
            socket.sendall(data_size.to_bytes(4, 'big'))  # Sending the size of data first
            socket.sendall(serialized_data)              # Sending the actual data   
        except Exception as e:
            print("Error in sending data",e)            # Sending the actual data   

    # Receive Data
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

    def evaluate(self, weights,w_id):
        print("Evaluating")
        state_dicts = weights
        unsorted_scores = self.model.eval(state_dicts,w_id)
        return unsorted_scores

    def update_model(self, avg_dicts):
        self.model.adapt_current_model(avg_dicts)

    def get_model_state_dict(self):
        return self.model.model.state_dict()

    def set_model_state_dict(self, state_dict):
        self.model.model.load_state_dict(state_dict)



    def get_model_uri(self):
        return self.contract_instance.functions.getModelURI().call()

    def get_round_number(self):
        return self.contract_instance.functions.getRound().call()


    def average(self, worker_weights):
        all_keys_list = list(worker_weights.keys())

        print("all keys: ", all_keys_list)

        averaged_weights = OrderedDict()
        for layer_key in worker_weights[all_keys_list[0]]:
            layer_weights = [worker_weights[worker][layer_key] for worker in worker_weights]
            averaged_weights[layer_key] = torch.stack(layer_weights).mean(dim=0)


        return averaged_weights
    
    
    
    def load_worker_data_from_json(self):
        # Load the worker data from JSON file
        with open('worker_data.json', 'r') as file:
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
    
    def connect_to_peer(self,ip, port):
        max_retries = 20  # Number of times to retry connecting
        retries = 0

        while retries < max_retries:
            try:
                client_socket_peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket_peer.connect((ip, int(port)))
                print("Connected to worker head peer")
                return client_socket_peer
            except ConnectionRefusedError:
                retries += 1
                print(f"Connection to peer {ip}:{port} failed. Retrying ({retries}/{max_retries})...")
                # Add a delay before retrying
                time.sleep(1)

        raise ConnectionError(f"Failed to connect to peer {ip}:{port} after {max_retries} retries.")
    

    def send_file(self, client_socket, file_path):
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                data = file.read()
                max_retries = 10  # Number of times to retry sending
                retries = 0

                while retries < max_retries:
                    try:
                        client_socket.sendall(data)
                        print("File sent successfully")
                        return True
                    except socket.error as e:
                        retries += 1
                        print(f"Failed to send file. Retrying ({retries}/{max_retries})...")
                        # Add a delay before retrying
                        time.sleep(1)

                raise ConnectionError(f"Failed to send file after {max_retries} retries.")
        else:
            return False

    def receive_file(self, server_socket, save_path):
        data = b""
        max_retries = 10  # Number of times to retry receiving
        retries = 0

        while retries < max_retries:
            try:
                while True:
                    chunk = server_socket.recv(1024)
                    if not chunk:
                        break
                    data += chunk

                if data:
                    with open(save_path, 'wb') as file:
                        file.write(data)
                        print("File received successfully")
                        return True
                else:
                    return False
            except socket.error as e:
                retries += 1
                print(f"Failed to receive file. Retrying ({retries}/{max_retries})...")
                # Add a delay before retrying
                time.sleep(1)

        raise ConnectionError(f"Failed to receive file after {max_retries} retries.")
    