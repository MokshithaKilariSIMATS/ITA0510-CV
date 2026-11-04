import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; im,gt=load_pair(25)
binary=binarize(im).astype(np.uint8)
for conn in [4,8]:
    n,lab,stats,cents=cv2.connectedComponentsWithStats(binary,connectivity=conn)
    rows.append({"method":f"connected_components_{conn}","regions":n-1,"mean_area":float(stats[1:,cv2.CC_STAT_AREA].mean()) if n>1 else 0,"IoU":iou(binary,gt)})
dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
sure=cv2.threshold(dist,0.35*dist.max(),255,0)[1].astype(np.uint8)
n,markers=cv2.connectedComponents(sure); markers=markers+1; unknown=cv2.subtract(np.uint8(binary*255),sure); markers[unknown==255]=0
color=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR); cv2.watershed(color,markers)
ws=(markers>1).astype(np.uint8)
rows.append({"method":"watershed","regions":int(len(np.unique(markers[markers>1]))),"mean_area":float(ws.sum()),"IoU":iou(ws,gt)})
save_preview(9,[im,binary*255,dist,ws*255],["Input","Binary regions","Distance map","Watershed"])
pd.DataFrame(rows).to_csv(out(9,"region_metrics.csv"),index=False)
