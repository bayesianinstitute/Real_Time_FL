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
        print("get",data.content)

        ipfs_data = data.content

        print(ipfs_data.decode('utf-8'))  # Assuming the data is in UTF-8 encoding
        print(data)
        # Handle the data (e.g., save it to a file)
        return data
    def get_from_ipfs(self, cid, save_path):
        params = (
            ('arg', cid),
        )
        infura_api_url = 'https://ipfs.infura.io:5001/api/v0/get'
        headers = {'Authorization': f'Bearer {self.project_secret}'}

        try:
            # Send a GET request to the Infura IPFS API with your API key
            response = requests.get(infura_api_url, params=params, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Save the retrieved data to a file
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print(f"File saved to {save_path}")
            else:
                print("Failed to retrieve data. Status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            print("Error:", e)

if __name__ == '__main__':
    # Create an instance of the IPFSInfuraConnection class
    ipfs_connection = IPFSInfuraConnection()

    # Specify the CID and the path where you want to save the downloaded file
    cid = "QmTgLJLEUwgUGHfmaZAoPqigCB2bFkAibZqcKLuy9vdgLN"
    save_path = "downloaded_file.txt"

    # Download the file from IPFS and save it to the specified path
    ipfs_connection.get_from_ipfs(cid, save_path)

