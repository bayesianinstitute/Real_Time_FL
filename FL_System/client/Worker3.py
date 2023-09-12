import torch
import socket
import ipfshttpclient
import torch
from collections import OrderedDict
import random
from Worker_Main import Worker
from config_app import HOST,PORT
import requests

def get_public_ip():
    try:
        response = requests.get('https://api64.ipify.org?format=json')
        if response.status_code == 200:
            data = response.json()
            return data['ip']
        else:
            return None
    except requests.RequestException:
        return None


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_evil = False
    topk = 1

    client_port = random.randint(40000, 50000)
    client_port_next = random.randint(50000, 60000)
    client_port_next_cluster=random.randint(50000, 60000)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    worker_dict = OrderedDict()
    worker_id = 3
    locations='INDIA'
    meta={
        'port':client_port_next,
        'locations':locations,
        'cluster_head_port':client_port_next_cluster
    }

    # Get the server's public IP address
    public_ip = get_public_ip()
    if public_ip is None:
        print("Unable to retrieve the server's public IP. Please check your internet connection.")
        exit()
    else:
        print("My  public IP: " + public_ip)

    # Reuse the socket address to avoid conflicts when restarting the program
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    public_ip='localhost'
    # Bind the worker's socket to the specified port
    client_socket.bind((public_ip, client_port))  # Bind to all available network interfaces

    client_socket.connect((HOST, PORT))
    print("Connected to server")
    current_port = client_socket.getsockname()[1]
    print("current port : ", current_port)
    key='0x7872991aaafc5252d2ca93c30765b0aa14ed1338aa259be5defb61872375cacb'
    worker = Worker( device, is_evil, topk,worker_id,key)

    # receive contract Address
    contract_address=worker.receive_data(client_socket)

    print("Contract address : ", contract_address)
    worker.join_task(contract_address)

    print("meta : ", meta)
    # Sending Meta data
    worker.send_data(client_socket, meta)

    # sending Worker Blockchain Address

    w_addr=worker.workerAddress()

    worker.send_data(client_socket, w_addr)    
    print("sent Address : ",w_addr)

    # Receive Json for Header
    received_json = worker.receive_data(client_socket)
    print("received_json : ", received_json)
    received_headid = worker.receive_data(client_socket)
    print("received_headid server : ", received_headid)
 
    results = []
    epoch = 0

    while True:
        epoch += 1
        workerAddress = worker.workerAddress()

        print("Training Model")
        print("received_headid : ", received_headid)

        weights = worker.train(round=1)

        accuracy,loss=worker.test()
        print('\nResult set: Accuracy:  ({:.0f}%), Loss: {:.6f}\n'.format(accuracy, loss))

        unsorted_scores =worker.evaluate(weights,worker_id)

        worker.send_data(client_socket, unsorted_scores)
        print("Send unscored scores")

        worker_index = received_headid['workerid']

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
                worker_weight = worker.receive_data(client_socket)
                worker.send_data(client_socket,"Ack")
                print("Receive worker-weights from client", idx + 1)
                worker_weights.append(worker_weight)


            # Assuming you want to store the worker addresses in the worker_dict
            for idx, weight in enumerate(worker_weights):
                # The key will be in the format 'worker_1_weights', 'worker_2_weights', and so on
                key = f'worker_{idx + 1}_weights'
                # Add the weight to the OrderedDict with the corresponding key
                worker_dict[key] = weight

            averaged_weights = worker.average(worker_dict)
            print("Averaged weights are Done")

                        # Now Suffle
            worker_head_id = worker.shuffle_worker_head(received_json)
            print("shuffle_id id ", worker_head_id)
            print("client_port_next_id ", client_port_next)


            old_client_port_next = client_port_next

            if worker_head_id != client_port_next:
                client_port_next = random.randint(50000, 60000)
                # Find the dictionary with 'workerid' equal to worker_head_id and update its 'new_port' value
                for entry in received_json:
                    if entry['workerid'] == worker_id:
                        entry['new_port'] = client_port_next
                        break

            received_headid = worker_head_id

            try:
                for idx, client_socket in enumerate(client_sockets):
                    print("Sending json file to client:", idx + 1)
                    worker.send_data(client_socket, worker_head_id)
                    worker.send_data(client_socket, received_json)
            
            except ConnectionResetError:
                # Handle the case when a client disconnects unexpectedly
                print("Client", idx + 1, "disconnected.")
                client_sockets.pop(idx)

            except Exception as e:
                print("Error sending data", e)

            worker.update_model(averaged_weights)
            print("Worker Update it works and adding weight to ipfs")

            model_filename = 'save_model/model_index_{}.pt'.format(worker_index)
            torch.save(averaged_weights, model_filename)
            print("MODEL SAVE TO LOCAL")

            try:
                for idx, client_socket in enumerate(client_sockets):
                    print("Sending model  to client:", idx + 1)
                    worker.send_file(client_socket,model_filename)
                    print("Sent ipfs hash to clients", idx + 1)

            except ConnectionResetError:
                # Handle the case when a client disconnects unexpectedly
                print("Client", idx + 1, "disconnected.")
                client_sockets.pop(idx)



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
                a=worker.receive_data(client_socket_peer)
                print(a)
                # Receive 'model.pt' file from the server
                received_headid = worker.receive_data(client_socket_peer)
                print("received_headid : ", received_headid)
                received_json = worker.receive_data(client_socket_peer)
                print("new received_json : ", received_json)




                try : 
                    worker.receive_file(client_socket, 'model.pt')
                    model_filename = 'save_model/model_index_{}.pt'.format(received_headid['workerid'])
                    average_Weight = torch.load(model_filename)
                    print("Received 'model.pt' file from the server and saved it as 'model_received.pt'.")
                except:
                    print("Failed to receive the 'model.pt' file from the server.")

                worker.update_model(average_Weight)
                print("Updated model weights")



  
                if received_headid['new_port'] == client_port_next:
                    print("I am the header again.")
                    is_header = True
                else:
                    is_header = False

            except Exception as e:
                print("Error during peer connection:", e)
    else :
        client_socket.close()
        print("Connection closed")