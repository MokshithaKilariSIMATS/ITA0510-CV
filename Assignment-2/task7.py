import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; im,gt=load_pair(25)
kernels={"point":np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.float32),"horizontal_line":np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]],np.float32),"vertical_line":np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]],np.float32)}
maps=[]
for name,k in kernels.items():
    m=cv2.filter2D(im,-1,k); edge=(m>100).astype(np.uint8); maps.append(edge)
    rows.append({"detector":name,"precision":float(((edge&gt)>0).sum()/(edge.sum()+1e-9)),"recall":float(((edge&gt)>0).sum()/(gt.sum()+1e-9))})
for name,m in [("Sobel",cv2.magnitude(cv2.Sobel(im,cv2.CV_32F,1,0),cv2.Sobel(im,cv2.CV_32F,0,1))>80),("Canny",cv2.Canny(im,60,140)>0)]:
    m=m.astype(np.uint8); rows.append({"detector":name,"precision":float(((m&gt)>0).sum()/(m.sum()+1e-9)),"recall":float(((m&gt)>0).sum()/(gt.sum()+1e-9))}); maps.append(m)
save_preview(7,maps,["Point","Horizontal line","Vertical line","Sobel","Canny"])
pd.DataFrame(rows).to_csv(out(7,"edge_detection_metrics.csv"),index=False)
