import ipfshttpclient

import os
from dotenv import load_dotenv
import requests
# Load environment variables from .env
load_dotenv()

project_id = os.getenv("PROJECT_ID")
project_secret = os.getenv("PROJECT_SECRET")

ipfs_hash = "QmTgLJLEUwgUGHfmaZAoPqigCB2bFkAibZqcKLuy9vdgLN"
# Create the Infura IPFS API URL
infura_api_url = f'https://ipfs.infura.io:5001/api/v0/cat?arg={ipfs_hash}'

try:
    # Send a GET request to the Infura IPFS API with your API key
    response = requests.get(infura_api_url, headers={'Authorization': f'Bearer {project_secret}'})

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # The content of the response contains the data from the IPFS hash
        ipfs_data = response.content

        # You can now use 'ipfs_data' as your retrieved data
        print(ipfs_data.decode('utf-8'))  # Assuming the data is in UTF-8 encoding
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
except requests.exceptions.RequestException as e:
    print("Error:", e)


# client = ipfshttpclient.connect()

# print("Connection established")


# import requests
# files = {
# 'file': 'Communication\ipfs\t1.txt'
# }
# response = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=files, auth=(project_id,project_secret))
# print(response["Hash"])
