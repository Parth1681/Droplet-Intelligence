"""Local website and prediction API. Binds to 127.0.0.1 by default."""
import argparse,json,io,zipfile
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
from threading import Lock
from .data import ROOT
from .predict import Predictor

predictor=Predictor();model_lock=Lock()
class Handler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        path=unquote(urlsplit(path).path).lstrip('/')
        base=ROOT if path.split('/')[0] in ['models','results','data'] else ROOT/'web'
        target=(base/(path or 'index.html')).resolve()
        if not target.is_relative_to(base.resolve()):return str(ROOT/'nonexistent')
        return str(target)
    def list_directory(self,path):self.send_error(403,'Directory listing disabled');return None
    def send_json(self,data,status=200):
        b=json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/api/health':return self.send_json({'status':'ok','version':predictor.release['version'],'model':predictor.release['selected']})
        if path=='/api/metadata':return self.send_json(predictor.release)
        if path=='/downloads/Droplet_Intelligence_v2.zip':
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in ROOT.rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts and 'sem_original' not in p.parts and not p.name.endswith(('.log','.zip')):z.write(p,'Droplet_Intelligence_v2/'+str(p.relative_to(ROOT)))
            content=b.getvalue();self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Disposition','attachment; filename="Droplet_Intelligence_v2.zip"');self.send_header('Content-Length',str(len(content)));self.end_headers();return self.wfile.write(content)
        return super().do_GET()
    def do_POST(self):
        if urlsplit(self.path).path!='/api/predict':return self.send_json({'error':'Unknown endpoint'},404)
        if self.headers.get_content_type()!='application/json':return self.send_json({'error':'Use application/json'},415)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>20000:raise ValueError('JSON body must be 1–20000 bytes')
            p=json.loads(self.rfile.read(length))
            with model_lock:result=predictor.predict(p)
            return self.send_json(result)
        except (ValueError,TypeError,KeyError,OverflowError) as e:return self.send_json({'error':str(e)},422)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);p.add_argument('--host',default='127.0.0.1');args=p.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),Handler);print(f'Droplet Intelligence: http://{args.host}:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
if __name__=='__main__':main()
