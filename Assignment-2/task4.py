import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for i in range(25,31):
    im,_=load_pair(i)
    methods={"unsharp":cv2.addWeighted(im,1.6,cv2.GaussianBlur(im,(0,0),2),-0.6,0),"laplacian":cv2.convertScaleAbs(im-cv2.Laplacian(im,cv2.CV_16S)),"high_boost":cv2.addWeighted(im,1.8,cv2.GaussianBlur(im,(5,5),1),-0.8,0)}
    for name,res in methods.items():
        edges=cv2.Canny(res,60,140); noise=float(res.std()); sharp=float(cv2.Laplacian(res,cv2.CV_64F).var())
        rows.append({"page":i,"method":name,"edge_pixels":int((edges>0).sum()),"sharpness_laplacian_var":sharp,"intensity_std":noise})
        if i==25: ims=[im]+list(methods.values())
save_preview(4,ims,["Original","Unsharp","Laplacian","High boost"])
pd.DataFrame(rows).to_csv(out(4,"sharpening_metrics.csv"),index=False)
