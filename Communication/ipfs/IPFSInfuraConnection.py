from ipfshttpclient import connect
import os
from dotenv import load_dotenv
import requests
# Load environment variables from .env
load_dotenv()

class IPFSInfuraConnection:
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        self.project_secret = os.getenv("PROJECT_SECRET")


    def add_to_ipfs(self, file):

        
        files = {
            'file': file,
            }
        response = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=files, auth=(self.project_id,self.project_secret))
        print(response.text)
        return response

  
        # print(f"File added to IPFS with CID: {res['Hash']}")

    def get_from_ipfs(self, cid):
        params = (
        ('arg',cid),
        )
        data = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params, auth=(self.project_id,self.project_secret))
        print(data)
        # Handle the data (e.g., save it to a file)
        return data

if __name__ == '__main__':

    # Create an instance of the IPFSInfuraConnection class
    ipfs_connection = IPFSInfuraConnection()

    # Example usage
    f1='Communication\ipfs\t1.txt'

    # ipfs_connection.add_to_ipfs(f1)
    ipfs_connection.get_from_ipfs("QmTgLJLEUwgUGHfmaZAoPqigCB2bFkAibZqcKLuy9vdgLN")

