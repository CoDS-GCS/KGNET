from http.server import BaseHTTPRequestHandler, HTTPServer
import json
class test:
    def show(self):
        return "aaaa"

class http_server:
    def __init__(self, t1):
        myHandler.t1 = t1
        server = HTTPServer(('', 6250), myHandler)
        server.serve_forever()

class myHandler(BaseHTTPRequestHandler):
    t1 = None
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
#         self._set_response()
        self.wfile.write(json.dumps({'semantic_affinity': self.t1.show()}).encode())
        return

class main:
    def __init__(self):
        self.t1 = test()

        self.server = http_server(self.t1)

if __name__ == '__main__':
    m = main()