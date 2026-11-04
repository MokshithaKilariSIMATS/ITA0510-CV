import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for i in range(25,31):
    im,gt=load_pair(i)
    for name,res in [("global_equalization",cv2.equalizeHist(im)),("CLAHE",cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(im))]:
        pred=binarize(res); rows.append({"page":i,"method":name,"local_contrast":float(res.std()),"SSIM_global":ssim_simple(im,res),"text_recall":float((pred&gt).sum()/(gt.sum()+1e-9))})
        if i==25: ims.append(res)
save_preview(2,[load_pair(25)[0]]+ims,["Input","Global HE","CLAHE"])
pd.DataFrame(rows).to_csv(out(2,"histogram_metrics.csv"),index=False)
