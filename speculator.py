from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sys

from mistral7b.mistral_mlx_scratch import load_model
import mlx.core as mx

class Speculator:
    def __init__(self, model_path):
        # here we'll initialize model and current search tree
        print(f"loading model from {model_path}")
        self.model, self.tokenizer = load_model(model_path)

    def speculate(self, prompt):
        x = mx.array(prompt)[None]
        logits, cache = self.model(x)
        y = [mx.argmax(logits[:, -1, :]).item()]
        return y

class SpeculatorHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, speculator, *args, **kwargs):
        print('SpeculatorHTTPHandler.__init__')
        self.speculator = speculator
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        req = json.loads(post_data.decode('utf-8'))
        res = {}
        if 'prompt' in req:
            res = {"next_tokens": self.speculator.speculate(req['prompt'])}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(res).encode())

    # TODO: this should be some health/stats
    def do_GET(self):
        print('do get')
        response_dict = {"message": self.speculator.speculate([1, 4222, 349, 264, 5565, 302, 28705])}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_dict).encode())


class SpeculatorHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, speculator):
        self.speculator = speculator
        super().__init__(server_address, RequestHandlerClass)
    
    def finish_request(self, request, client_address):
        # Pass db_connection to the handler
        self.RequestHandlerClass(self.speculator, request, client_address, self)


if __name__ == '__main__':
    addr = 'localhost'
    port = 8808
    spec = Speculator(sys.argv[1])

    server_address = (addr, port)
    
    httpd = SpeculatorHTTPServer(server_address, SpeculatorHTTPHandler, spec)
    print(f"Speculator server started on http://{addr}:{port}")
    httpd.serve_forever()        

