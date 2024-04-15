# let's test access to speculator first

import requests
import json

def send_request():
    addr = 'localhost'
    port = 8808

    url = f'http://{addr}:{port}'
    
    data = {'prompt': [1, 4222, 349, 264, 5565, 302, 28705]}
    
    response = requests.post(url, json=data)
    received_data = response.json()
    
    print('Response from server:', received_data)

# TODO: this should be also a service, which connects to the speculator
# this service would wait for queries with some typical API, queue them and run model
# we can also dump kv cache somewhere on disk?

def speculation_loop():
    # here we get something like 
    # model, tokenizer
    # current tree to evaluate
    # function to get next tree (which would call speculator)

    while True:
        # if we have query, ...

        pass

        # can we send delta if it is same loop?
        # speculator.get(query_id, curr) 
    
    pass


if __name__ == '__main__':
    send_request()