import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for i in range(25,31):
    im,gt=load_pair(i)
    methods={"fixed":cv2.threshold(im,150,255,cv2.THRESH_BINARY_INV)[1],"otsu":cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1],"adaptive":cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)}
    for name,res in methods.items():
        pred=(res>0).astype(np.uint8); rows.append({"page":i,"method":name,"Dice":dice(pred,gt),"IoU":iou(pred,gt)})
        if i==25: ims.append(res)
save_preview(8,ims,["Fixed","Otsu","Adaptive"])
pd.DataFrame(rows).to_csv(out(8,"threshold_metrics.csv"),index=False)
