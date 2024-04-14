# let's test access to speculator first

import requests
import json

def send_request():
    addr = 'localhost'
    port = 8808

    url = f'http://{addr}:{port}'
    
    # Data to be sent to server
    data = {'prompt': [1, 4222, 349, 264, 5565, 302, 28705]}
    
    # Send POST request with prompt
    response = requests.post(url, json=data)
    
    # Convert response to JSON
    received_data = response.json()
    print('Response from server:', received_data)

if __name__ == '__main__':
    send_request()