import glob,os,subprocess,time,json,statistics
SRC='/mnt/SD_Dataset/训练器相关/krea2-webui/image_dataset/style_8_4'
allfs=sorted(glob.glob(SRC+'/*.jpg'))
def pick(lo,hi,cap):
    g=[f for f in allfs if lo<=os.path.getsize(f)/1e6<hi]
    if len(g)<=cap: return g
    st=len(g)/cap; return [g[int(i*st)] for i in range(cap)]
SAMP={'1-5MB':pick(1,5,2),'5-10MB':pick(5,10,2),'10-15MB':pick(10,15,10)}
def run(cmd):
    t=time.perf_counter(); p=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    peak=0
    while True:
        pid,st,ru=os.wait4(p.pid,os.WNOHANG)
        if pid: break
        try:
            for l in open(f'/proc/{p.pid}/status'):
                if l.startswith('VmHWM'): peak=max(peak,int(l.split()[1])); break
        except Exception: pass
        time.sleep(0.005)
    wall=time.perf_counter()-t
    return wall,ru.ru_utime+ru.ru_stime,max(peak,ru.ru_maxrss)/1024.0
JPE=['JPEG q95 4:4:4',95,0],['JPEG q92 4:4:4',92,0]
VARS=[('JPEG q95 4:4:4',"raise SystemExit()"),]
out=open('/tmp/bm2/cpu.jsonl','w',buffering=1)
for bucket,fs in SAMP.items():
    for f in fs:
        for name,q,ss in [('JPEG q95 4:4:4',95,0),('JPEG q92 4:4:4',92,0),('JPEG q95 4:2:0',95,2)]:
            code=f"from PIL import Image\nim=Image.open({f!r}); im.load(); im.convert('RGB').save('/tmp/bm2/c.jpg',format='JPEG',quality={q},subsampling={ss})"
            w,c,m=run(['/usr/bin/python3','-c',code])
            out.write(json.dumps(dict(bucket=bucket,name=name,op='enc',wall=w,cpu=c,rss=m))+"\n")
            code2="from PIL import Image\nim=Image.open('/tmp/bm2/c.jpg'); im.load(); im.convert('RGB').tobytes()"
            w,c,m=run(['/usr/bin/python3','-c',code2])
            out.write(json.dumps(dict(bucket=bucket,name=name,op='dec',wall=w,cpu=c,rss=m))+"\n")
        for name,args in [('JXL 无损重压缩',['-d','0','-e','8']),('JXL -d 0.5',['-d','0.5','-e','7','--lossless_jpeg=0']),('JXL -d 1.0',['-d','1','-e','7','--lossless_jpeg=0'])]:
            w,c,m=run(['cjxl',f,'/tmp/bm2/c.jxl']+args)
            out.write(json.dumps(dict(bucket=bucket,name=name,op='enc',wall=w,cpu=c,rss=m))+"\n")
            w,c,m=run(['djxl','/tmp/bm2/c.jxl','/tmp/bm2/c.ppm'])
            out.write(json.dumps(dict(bucket=bucket,name=name,op='dec',wall=w,cpu=c,rss=m))+"\n")
    print(bucket,'done',flush=True)
