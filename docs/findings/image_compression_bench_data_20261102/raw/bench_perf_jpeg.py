import glob,os,time,statistics,json
from PIL import Image
SRC='/mnt/SD_Dataset/训练器相关/krea2-webui/image_dataset/style_8_4'
allfs=sorted(glob.glob(SRC+'/*.jpg'))
def pick(lo,hi,cap):
    g=[f for f in allfs if lo<=os.path.getsize(f)/1e6<hi]
    if len(g)<=cap: return g
    st=len(g)/cap; return [g[int(i*st)] for i in range(cap)]
BUCKETS=[('1-5MB',pick(1,5,5)),('5-10MB',pick(5,10,5)),('10-15MB',pick(10,15,10))]
VAR=[('JPEG q98 4:4:4',98,0),('JPEG q95 4:4:4',95,0),('JPEG q92 4:4:4',92,0),('JPEG q95 4:2:0',95,2),('JPEG q92 4:2:0',92,2)]
out=open('/tmp/bm2/perf_jpeg.jsonl','w',buffering=1)
# warm up caches
_=Image.open(allfs[0]); _.load()
for bucket,fs in BUCKETS:
    for f in fs:
        im=Image.open(f); im.load(); rgb=im.convert('RGB')
        for name,q,ss in VAR:
            ts=[]
            for _ in range(2):   # 取第 2 次，避开首轮写盘冷缓存
                t=time.perf_counter(); rgb.save('/tmp/bm2/p.jpg',format='JPEG',quality=q,subsampling=ss); ts.append(time.perf_counter()-t)
            e=min(ts)
            ts=[]
            for _ in range(2):
                t=time.perf_counter()
                d=Image.open('/tmp/bm2/p.jpg'); d.load(); _arr=d.convert('RGB').tobytes(); ts.append(time.perf_counter()-t)
            dv=min(ts)
            out.write(json.dumps(dict(bucket=bucket,name=name,enc=e,dec=dv,orig=os.path.getsize(f)))+'\n')
    print(bucket,'done',flush=True)
