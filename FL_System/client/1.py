
import requests
import csv
import time

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

    # Get the server's public IP address
public_ip = get_public_ip()
if public_ip is None:
        print("Unable to retrieve the server's public IP. Please check your internet connection.")
        exit()
else:
        print("My  public IP: " + public_ip)
