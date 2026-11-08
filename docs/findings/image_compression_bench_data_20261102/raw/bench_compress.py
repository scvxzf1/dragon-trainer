import glob, os, subprocess, time, statistics, json, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SRC='/mnt/SD_Dataset/训练器相关/krea2-webui/image_dataset/style_8_4'
allfs=sorted(glob.glob(SRC+'/*.jpg'))
def pick(lo,hi,cap):
    g=[f for f in allfs if lo<=os.path.getsize(f)/1e6<hi]
    if len(g)<=cap: return g
    step=len(g)/cap
    return [g[int(i*step)] for i in range(cap)]
BUCKETS=[('1-5MB',pick(1,5,5)),('5-10MB',pick(5,10,5)),('10-15MB',pick(10,15,10))]

# ---------- perf helpers ----------
def run_child(cmd):
    t=time.perf_counter()
    p=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    peak=0
    while p.poll() is None:
        try:
            for line in open(f'/proc/{p.pid}/status'):
                if line.startswith('VmHWM'):
                    peak=max(peak,int(line.split()[1])); break
        except Exception: pass
        time.sleep(0.008)
    p.wait(); wall=time.perf_counter()-t
    ru=os.wait4(p.pid,0)[2] if False else None
    return wall, peak
BASE_WALL, BASE_RSS = 0.055, 32*1024   # python -c 基线估计
def run_py(code):
    t=time.perf_counter()
    r=subprocess.run(['/usr/bin/python3','-c',code],capture_output=True)
    if r.returncode!=0: raise RuntimeError(r.stderr.decode()[-400:])
    wall=time.perf_counter()-t
    return wall
# measure baseline properly
b=sorted([run_py('pass') for _ in range(5)])[2]

# ---------- metrics ----------
def arr(im): return np.asarray(im.convert('RGB'), dtype=np.float32)
def srgb2lab(rgb):
    c=rgb/255.0
    c=np.where(c<=0.04045, c/12.92, ((c+0.055)/1.055)**2.4)
    M=np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
    xyz=c@M.T/np.array([0.95047,1.0,1.08883])
    f=np.where(xyz>0.008856, np.cbrt(xyz), 7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])],-1)
class Ref:
    def __init__(self, rgb):
        self.rgb=rgb; self.lab=srgb2lab(rgb)
        self.ps=[]; self.sref=[]
        for c in range(3):
            a=rgb[...,c]; m=gaussian_filter(a,1.5,mode='reflect'); v=gaussian_filter(a*a,1.5,mode='reflect')-m*m
            self.sref.append((m,np.maximum(v,0)))
    def psnr(self,b):
        return [10*np.log10(255.0**2/max(((self.rgb[...,c]-b[...,c])**2).mean(),1e-9)) for c in range(3)]
    def ssim(self,b):
        C1,C2=(0.01*255)**2,(0.03*255)**2; out=[]
        for c in range(3):
            m1,v1=self.sref[c]; a=b[...,c]
            m2=gaussian_filter(a,1.5,mode='reflect'); v2=np.maximum(gaussian_filter(a*a,1.5,mode='reflect')-m2*m2,0)
            c12=gaussian_filter(a*b[...,c]*0+ (self.rgb[...,c]*a),1.5,mode='reflect')-m1*m2
            out.append(float((((2*m1*m2+C1)*(2*c12+C2))/((m1*m1+m2*m2+C1)*(v1+v2+C2))).mean()))
        return out
    def de76(self,b):
        d=np.sqrt(((self.lab-srgb2lab(b))**2).sum(-1))
        return float(d.mean()), float(np.percentile(d,95))

VARIANTS=[('JPEG q98 4:4:4','jpeg',dict(quality=98,subsampling=0)),
          ('JPEG q95 4:4:4','jpeg',dict(quality=95,subsampling=0)),
          ('JPEG q92 4:4:4','jpeg',dict(quality=92,subsampling=0)),
          ('JPEG q95 4:2:0','jpeg',dict(quality=95,subsampling=2)),
          ('JPEG q92 4:2:0','jpeg',dict(quality=92,subsampling=2)),
          ('JXL 无损重压缩','jxl0',['-d','0','-e','8']),
          ('JXL -d 0.5','jxl',['-d','0.5','-e','7','--lossless_jpeg=0']),
          ('JXL -d 1.0','jxl',['-d','1','-e','7','--lossless_jpeg=0'])]

JL=open('/tmp/bm2/results.jsonl','a',buffering=1)
def measure(f):
    im=Image.open(f); im.load(); ref=Ref(arr(im))
    orig=os.path.getsize(f); rows=[]
    for name,kind,kw in VARIANTS:
        enc=[]
        if kind=='jpeg':
            code=f"""from PIL import Image
im=Image.open({f!r}); im.load(); im.convert('RGB').save('/tmp/bm2/o.jpg',format='JPEG',quality={kw['quality']},subsampling={kw['subsampling']})"""
            w=run_py(code)-b; sz=os.path.getsize('/tmp/bm2/o.jpg')
            dcode=f"""from PIL import Image
im=Image.open('/tmp/bm2/o.jpg'); im.load(); im.convert('RGB').tobytes()"""
            dw=run_py(dcode)-b
            d=Image.open('/tmp/bm2/o.jpg'); d.load(); img=arr(d); out='/tmp/bm2/o.jpg'
        else:
            args=['cjxl',f,'/tmp/bm2/o.jxl']+kw
            w2,_=run_child(args); w=w2
            sz=os.path.getsize('/tmp/bm2/o.jxl')
            dw2,_=run_child(['djxl','/tmp/bm2/o.jxl','/tmp/bm2/o.ppm'])
            dw=dw2
            img=arr(Image.open('/tmp/bm2/o.ppm')); out='/tmp/bm2/o.jxl'
        p=ref.psnr(img); s=ref.ssim(img); de=ref.de76(img)
        rows.append(dict(bucket=BUCKET,f=os.path.basename(f),name=name,bytes=sz,orig=orig,enc=w,dec=dw,
                         psnr=float(np.mean(p)),psnr_r=float(p[0]),psnr_g=float(p[1]),psnr_b=float(p[2]),
                         ssim=float(np.mean(s)),de=de[0],de95=de[1]))
    for r in rows: JL.write(json.dumps(r)+'\n')
    return rows

BUCKET=None
for label,fs in BUCKETS:
    BUCKET=label
    print(f"### {label} n={len(fs)}", flush=True)
    agg={}
    for i,f in enumerate(fs):
        for r in measure(f):
            a=agg.setdefault(r['name'],{'b':0,'o':0,'p':[],'pr':[],'pg':[],'pb':[],'s':[],'de':[],'de95':[],'e':[],'d':[]})
            a['b']+=r['bytes']; a['o']+=r['orig']
            a['p'].append(r['psnr']); a['pr'].append(r['psnr_r']); a['pg'].append(r['psnr_g']); a['pb'].append(r['psnr_b'])
            a['s'].append(r['ssim']); a['de'].append(r['de']); a['de95'].append(r['de95'])
            a['e'].append(r['enc']); a['d'].append(r['dec'])
        print(f'  [{i+1}/{len(fs)}] {os.path.basename(f)[:30]}', flush=True)
    out=[]
    for name,a in agg.items():
        out.append(dict(name=name,tot=a['b']/1e6,orig=a['o']/1e6,save=100*(1-a['b']/a['o']),bw=a['o']/a['b'],
                        psnr=statistics.mean(a['p']),pr=statistics.mean(a['pr']),pg=statistics.mean(a['pg']),pb=statistics.mean(a['pb']),
                        ssim=statistics.mean(a['s']),de=statistics.mean(a['de']),de95=statistics.mean(a['de95']),
                        enc=statistics.mean(a['e']),dec=statistics.mean(a['d']),
                        enc_mbs=(a['o']/1e6)/sum(a['e']),dec_mbs=(a['o']/1e6)/sum(a['d']),n=len(a['s'])))
    out.sort(key=lambda r:-r['psnr'])
    print(f'{"方案":<18}{"总MB":>8}{"省%":>7}{"带宽×":>7}{"PSNR":>7}{"SSIM":>8}{"ΔE76":>7}{"ΔE95":>7}{"编s":>7}{"解s":>7}')
    for r in out:
        print(f'{r["name"]:<18}{r["tot"]:>8.1f}{r["save"]:>7.1f}{r["bw"]:>7.2f}{r["psnr"]:>7.2f}{r["ssim"]:>8.4f}{r["de"]:>7.2f}{r["de95"]:>7.2f}{r["enc"]:>7.3f}{r["dec"]:>7.3f}')
    json.dump(out,open(f'/tmp/bm2/res_{label.split()[0]}.json','w'))
