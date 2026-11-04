import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for i in range(25,31):
    im,gt=load_pair(i); F=np.fft.fftshift(np.fft.fft2(im)); h,w=im.shape; yy,xx=np.ogrid[:h,:w]; d=np.sqrt((yy-h/2)**2+(xx-w/2)**2)
    for family in ["ideal","butterworth","gaussian"]:
        cutoff=22
        if family=="ideal": H=(d>=cutoff).astype(float)
        elif family=="gaussian": H=1-np.exp(-(d*d)/(2*cutoff*cutoff))
        else: H=1-1/(1+(d/(cutoff+1e-9))**4)
        res=np.uint8(np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(F*H))),0,255))
        pred=(res>35).astype(np.uint8)
        rows.append({"page":i,"filter":family,"edge_contrast":float(res.std()),"background_noise":float(res[gt==0].std()),"text_recall":float((pred&gt).sum()/(gt.sum()+1e-9))})
        if i==25: ims.append(cv2.normalize(res,None,0,255,cv2.NORM_MINMAX))
save_preview(6,ims,["Ideal HP","Butterworth HP","Gaussian HP"])
pd.DataFrame(rows).to_csv(out(6,"highpass_metrics.csv"),index=False)
