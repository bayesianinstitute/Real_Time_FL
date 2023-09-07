import ipfshttpclient

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

project_id = os.getenv("PROJECT_ID")
project_secret = os.getenv("PROJECT_SECRET")

client = ipfshttpclient.connect()

print("Connection established")


import requests
files = {
'file': 'Communication\ipfs\t1.txt'
}
response = requests.post('https://ipfs.infura.io:5001/api/v0/add', files=files, auth=(project_id,project_secret))
print(response["Hash"])
