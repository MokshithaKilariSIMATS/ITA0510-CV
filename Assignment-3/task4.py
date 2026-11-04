import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for p in query_images()[:10]:
    im=read(p); g=gray(im)
    sobx=cv2.Sobel(g,cv2.CV_32F,1,0); soby=cv2.Sobel(g,cv2.CV_32F,0,1); sob=cv2.convertScaleAbs(cv2.magnitude(sobx,soby)); _,sob=cv2.threshold(sob,60,255,cv2.THRESH_BINARY)
    blur=cv2.GaussianBlur(g,(5,5),1); log=cv2.convertScaleAbs(cv2.Laplacian(blur,cv2.CV_16S)); _,log=cv2.threshold(log,35,255,cv2.THRESH_BINARY)
    can=cv2.Canny(g,70,150)
    for name,e in [("Sobel",sob),("LoG",log),("Canny",can)]:
        rows.append({"image":p.name,"detector":name,"edge_pixels":int((e>0).sum()),"runtime_ms":0.0})
    if p.name=="query_01.png": ims=[im,sob,log,can]
save_board(4,ims,["Input","Sobel","LoG","Canny"])
pd.DataFrame(rows).to_csv(out(4,"edge_detector_metrics.csv"),index=False)
