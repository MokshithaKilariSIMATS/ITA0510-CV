import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rng=np.random.default_rng(7); rows=[]; ims=[]
for i in range(25,31):
    clean,_=load_pair(i)
    noise=rng.normal(0,12,clean.shape); noisy=np.clip(clean.astype(float)+noise,0,255).astype(np.uint8)
    filters=[("Gaussian_3",cv2.GaussianBlur(noisy,(3,3),0)),("Median_3",cv2.medianBlur(noisy,3)),("Bilateral",cv2.bilateralFilter(noisy,7,45,45))]
    for name,res in filters:
        rows.append({"page":i,"filter":name,"PSNR_dB":psnr(clean,res),"SSIM_global":ssim_simple(clean,res),"stroke_recall":float((binarize(res)&binarize(clean)).sum()/(binarize(clean).sum()+1e-9))})
        if i==25: ims=[clean,noisy]+[x[1] for x in filters]
save_preview(3,ims,["Clean","Noisy","Gaussian","Median","Bilateral"])
pd.DataFrame(rows).to_csv(out(3,"denoising_metrics.csv"),index=False)
