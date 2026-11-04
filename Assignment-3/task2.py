import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; preview=[]
for q in query_images()[:20]:
    im=read(q); g=gray(im)
    a=cv2.resize(g,(256,360))
    b=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(cv2.medianBlur(a,3))
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); v=hsv[:,:,2]; glare=(v>245).astype(np.uint8)*255
    c=cv2.inpaint(b,glare,3,cv2.INPAINT_TELEA) if glare.shape==b.shape else b
    for name,res in [("minimal",a),("noise_contrast",b),("full_normalization",c)]:
        kp,_=cv2.ORB_create(nfeatures=600).detectAndCompute(res,None)
        rows.append({"query":q.name,"pipeline":name,"keypoints":len(kp),"processing_ms_estimate":0.0})
    if q.name=="query_01.png": preview=[g,a,b,c]
save_board(2,preview,["Original","Minimal","Denoise + CLAHE","Full normalization"])
pd.DataFrame(rows).to_csv(out(2,"preprocessing_comparison.csv"),index=False)
