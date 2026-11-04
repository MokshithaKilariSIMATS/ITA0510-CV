import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
db=db_images(); rows=[]; examples=[]
for p in db:
    im=read(p); g=gray(im); pix=resize_gray(im).flatten().astype(np.float32)
    hist=cv2.calcHist([im],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]); cv2.normalize(hist,hist)
    kp,des=orb_desc(im)
    rows.append({"product":p.name,"pixel_vector_bytes":pix.nbytes,"hsv_histogram_bytes":hist.nbytes,"orb_descriptor_bytes":0 if des is None else des.nbytes,"orb_keypoints":len(kp)})
    if len(examples)<3: examples.append(im)
pd.DataFrame(rows).to_csv(out(3,"representation_storage.csv"),index=False)
save_board(3,examples,["Packet A","Packet B","Packet C"])
