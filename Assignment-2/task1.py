import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; imgs=[]
for i in range(25,31):
    im,gt=load_pair(i)
    methods={"linear":cv2.normalize(im,None,0,255,cv2.NORM_MINMAX),"gamma":np.uint8(255*(im/255.0)**0.75),"log":np.uint8(255*np.log1p(im)/np.log(256))}
    for name,res in methods.items():
        contrast=float(res.std()); hist=cv2.calcHist([res],[0],None,[256],[0,256]).ravel(); prob=hist/(hist.sum()+1e-9); ent=float(-(prob*np.log2(prob+1e-12)).sum())
        pred=(res<150).astype(np.uint8); rows.append({"page":i,"method":name,"contrast_std":contrast,"entropy_bits":ent,"text_recall":float((pred&gt).sum()/(gt.sum()+1e-9))})
        if i==25: imgs.append(res)
save_preview(1,[load_pair(25)[0]]+imgs,["Original","Linear","Gamma","Log"])
pd.DataFrame(rows).to_csv(out(1,"point_transform_metrics.csv"),index=False)
