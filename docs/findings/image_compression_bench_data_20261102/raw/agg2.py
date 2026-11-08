import json, statistics, collections
rows=[json.loads(l) for l in open('/tmp/bm2/results.jsonl')]
perf={}
for l in open('/tmp/bm2/perf_jpeg.jsonl'):
    d=json.loads(l); perf.setdefault((d['bucket'],d['name']),[]).append(d)
by=collections.defaultdict(list)
for r in rows: by[(r['bucket'],r['name'])].append(r)
for (b,n),v in perf.items():
    if (b,n) in by:
        t={x['f']:x for x in by[(b,n)]}
        for d in v:
            pass
ORDER=['JXL 无损重压缩','JPEG q98 4:4:4','JPEG q95 4:4:4','JPEG q92 4:4:4','JPEG q95 4:2:0','JPEG q92 4:2:0','JXL -d 0.5','JXL -d 1.0']
def tbl(bucket):
    out=[]
    for name in ORDER:
        v=by.get((bucket,name))
        if not v: continue
        o=sum(x['orig'] for x in v); b=sum(x['bytes'] for x in v)
        pv=perf.get((bucket,name))
        if pv: enc=statistics.mean(x['enc'] for x in pv); dec=statistics.mean(x['dec'] for x in pv); src='in-process'
        else:  enc=statistics.mean(x['enc'] for x in v); dec=statistics.mean(x['dec'] for x in v); src='CLI'
        out.append(dict(name=name,n=len(v),orig=o/1e6,out=b/1e6,save=100*(1-b/o),bw=o/b,
            psnr=statistics.mean(x['psnr'] for x in v),pr=statistics.mean(x['psnr_r'] for x in v),
            pg=statistics.mean(x['psnr_g'] for x in v),pb=statistics.mean(x['psnr_b'] for x in v),
            ssim=statistics.mean(x['ssim'] for x in v),de=statistics.mean(x['de'] for x in v),
            de95=statistics.mean(x['de95'] for x in v),enc=enc,dec=dec,src=src,
            enc_mbs=(o/1e6)/sum(x['enc'] for x in (pv or v)),dec_mbs=(o/1e6)/sum(x['dec'] for x in (pv or v))))
    return out
json.dump({b:tbl(b) for b in ['1-5MB','5-10MB','10-15MB']}, open('/tmp/bm2/final.json','w'))
for bucket in ['1-5MB','5-10MB','10-15MB']:
    t=tbl(bucket)
    if not t: continue
    print(f'\n#### {bucket}（n={t[0]["n"]}，源 {t[0]["orig"]:.1f} MB，均 {t[0]["orig"]/t[0]["n"]:.2f} MB/张）\n')
    print('| 方案 | 压后 | 省 | 带宽× | PSNR-RGB | R/G/B | SSIM | ΔE76 | ΔE95 | 编码 s/张 | 解码 s/张 | 编码吞吐 | 解码吞吐 |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in t:
        print(f'| {r["name"]} | {r["out"]:.1f} MB | {r["save"]:.1f}% | {r["bw"]:.2f}× | {r["psnr"]:.2f} | {r["pr"]:.1f}/{r["pg"]:.1f}/{r["pb"]:.1f} | {r["ssim"]:.4f} | {r["de"]:.2f} | {r["de95"]:.2f} | {r["enc"]:.3f} | {r["dec"]:.3f} | {r["enc_mbs"]:.1f} MB/s | {r["dec_mbs"]:.1f} MB/s |')
