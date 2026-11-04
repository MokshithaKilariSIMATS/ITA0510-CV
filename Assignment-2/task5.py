import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for i in range(25,31):
    im,_=load_pair(i); F=np.fft.fftshift(np.fft.fft2(im)); h,w=im.shape; yy,xx=np.ogrid[:h,:w]; d=np.sqrt((yy-h/2)**2+(xx-w/2)**2)
    for family in ["ideal","butterworth","gaussian"]:
        for cutoff in [18,35]:
            if family=="ideal": H=(d<=cutoff).astype(float)
            elif family=="gaussian": H=np.exp(-(d*d)/(2*cutoff*cutoff))
            else: H=1/(1+(d/(cutoff+1e-9))**4)
            res=np.uint8(np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(F*H))),0,255))
            rows.append({"page":i,"filter":family,"cutoff":cutoff,"PSNR_dB":psnr(im,res),"SSIM_global":ssim_simple(im,res),"edge_blur":float(abs(cv2.Laplacian(im,cv2.CV_64F).var()-cv2.Laplacian(res,cv2.CV_64F).var()))})
            if i==25 and cutoff==18: ims.append(res)
save_preview(5,[load_pair(25)[0]]+ims,["Original","Ideal","Butterworth","Gaussian"])
pd.DataFrame(rows).to_csv(out(5,"lowpass_metrics.csv"),index=False)
