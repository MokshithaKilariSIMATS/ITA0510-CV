import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for p in query_images()[:10]:
    im=read(p); g=gray(im)
    h=cv2.cornerHarris(np.float32(g),2,3,.04); hmap=(h>0.01*h.max()).astype(np.uint8)*255
    shi=cv2.goodFeaturesToTrack(g,100,.01,8); shi_img=im.copy()
    if shi is not None:
        for x,y in np.int32(shi).reshape(-1,2): cv2.circle(shi_img,(x,y),3,(0,0,255),-1)
    fast=cv2.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True); k=fast.detect(g,None); fast_img=cv2.drawKeypoints(im,k,None,color=(0,255,0))
    rows += [{"image":p.name,"detector":"Harris","corners":int((hmap>0).sum())},{"image":p.name,"detector":"Shi-Tomasi","corners":0 if shi is None else len(shi)},{"image":p.name,"detector":"FAST","corners":len(k)}]
    if p.name=="query_01.png": ims=[hmap,shi_img,fast_img]
save_board(5,ims,["Harris response","Shi-Tomasi","FAST"])
pd.DataFrame(rows).to_csv(out(5,"corner_detector_metrics.csv"),index=False)
