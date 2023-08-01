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
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class Model():
    '''
    Contains all machine learning functionality
    '''
    # static, might have to be calculated dynamically
    batch_size = 64
    epochs = 1

    def __init__(self,  model, optimizer, device, topk, isEvil = False):
        self.num_workers = 1
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
        print('\nTest set: , Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, self.num_test_batches * self.batch_size,
          100. * correct / (self.num_test_batches * self.batch_size)))
        
        return correct / (self.num_test_batches*self.batch_size)
    
    def eval(self, model_state_dicts):
        res = []
        for idx, m in enumerate(model_state_dicts):
            self.model.load_state_dict(m)
            acc = self.test()
            res.append((acc,idx,m))
            
            
        sorted_models = sorted(res, key=lambda t: t[0])
        return self.rank_models(sorted_models),  self.get_top_k(sorted_models), res
            
            
            

class Worker:
    truffle_file = json.load(open('./build/contracts/FLTask.json'))

    

    def __init__(self, ipfs_path, device, is_evil, topk,worker_id):
        # self.bcc = BCCommunicator()
        # self.fsc = FSCommunicator(ipfs_path, device)


                        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        # Serialize public key to PEM format
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Generate AES key
        self.aes_key = Fernet.generate_key()
        print("ase", self.aes_key)

        # Create Fernet object with AES key
        self.fernet = Fernet(self.aes_key)

        self.key = os.getenv(f'WORKER{worker_id}_KEY')
        self.client_url = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')


        self.model_hash = 'QmZaeFLUPJZopTvKWsuji2Q5RPWaTLBRtpCoBbDM6sDyqM'
        model_bytes = self.client_url.cat(self.model_hash)
        model = torch.jit.load(io.BytesIO(model_bytes),
                               map_location=device)
        print("Done Model!!!!!!")

        optimizer_hash = 'Qmd96G9irL6hQuSfGFNoYqeVgg8DvAyv6GCt9CqCsEDj1w'
        optimizer_bytes = self.client_url.cat(optimizer_hash)
        opt = torch.load(io.BytesIO(
            optimizer_bytes), map_location=device)
        print("Done Optimizer !!!!!")



        # model, opt = self.fsc.fetch_initial_model()
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

        key = '0x36b2bb51e03ff1a6e27ff7f9054fbcce5d20d41813838028973a4ae3759a0026'
        self.w3 = Web3(HTTPProvider("http://localhost:7545"))
        if self.w3.isConnected():
            print("Worker initialization: connected to blockchain")

        self.account = self.w3.eth.account.privateKeyToAccount(key)
        self.contract = self.w3.eth.contract(
            bytecode=self.truffle_file['bytecode'], abi=self.truffle_file['abi'])



 
    def join_task(self):
        contract_address = '0xA2a3E91f54f60b62EF72BAa3b03d20E23F46Dee2'  # Double-check this address
        self.contract_instance = self.w3.eth.contract(
            abi=self.truffle_file['abi'], address=contract_address)


        # print("contract_instace",self.contract_instance)
        # Call the 'joinTask' function
        try:
            tx = self.contract_instance.functions.joinTask().buildTransaction({
                "gasPrice": self.w3.eth.gas_price,
                "chainId": 1337,
                "from": self.account.address,
                "nonce": self.w3.eth.getTransactionCount(self.account.address)
            })
            signed_tx = self.w3.eth.account.signTransaction(tx, self.key)
            tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)

            # Wait for the transaction to be mined and get the receipt
            tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)

            # Check if the transaction was successful
            if tx_receipt['status'] == 1:
                print("Transaction successful!")
            else:
                print("Transaction failed!")

        except Exception as e:
            print("Error while executing the transaction:", e)


    def workerAddress(self):
        print("Addressing: ", self.account.address)
        return self.account.address

    def train(self, round):
        # Train the model
        cur_state_dict = self.model.train()
        print("Modle Summary",self.model)
        # print("Training state: ", cur_state_dict)
        return cur_state_dict
    
    # def test(self):

    #     return self.model.test()


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
    

    def evaluate(self, weights):
        print("Evaluating")
        state_dicts = weights
        ranks, topk_dicts, unsorted_scores = self.model.eval(state_dicts)
        # topk_dicts.append(self.model.model.state_dict())
        # return self.model.average(topk_dicts), topk_dicts, unsorted_scores

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
        max_retries = 5  # Number of times to retry connecting
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
