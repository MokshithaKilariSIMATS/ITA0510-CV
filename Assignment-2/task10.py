import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; previews=[]
def scanner(image):
    # Enhancement, page boundary approximation, deskew placeholder and text segmentation.
    enhanced=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(image)
    edges=cv2.Canny(enhanced,50,150); contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    page=enhanced.copy()
    if contours:
        c=max(contours,key=cv2.contourArea); x,y,w,h=cv2.boundingRect(c); page=enhanced[y:y+h,x:x+w]
    binary=cv2.adaptiveThreshold(page,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)
    return page,binary
for i in range(25,31):
    im,gt=load_pair(i); t=time.perf_counter(); page,seg=scanner(im); elapsed=time.perf_counter()-t
    pred=cv2.resize((seg>0).astype(np.uint8),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
    rows.append({"page":i,"Dice":dice(pred,gt),"IoU":iou(pred,gt),"runtime_ms":elapsed*1000,"manual_review":bool(dice(pred,gt)<0.45)})
    if i==25: previews=[im,page,seg]
save_preview(10,previews,["Input page","Enhanced crop","Segmented text"])
pd.DataFrame(rows).to_csv(out(10,"scanner_quality_metrics.csv"),index=False)
with open(out(10,"deployment_report.txt"),"w") as f:
    f.write("CO2 Smart Document Scanner - synthetic demonstration report\\n")
    f.write("Test split: pages 25-30. Dataset: 30 parameterized synthetic page images with generated text/diagram masks.\\n")
    f.write("Selected baseline: CLAHE + adaptive Gaussian thresholding. Review pages flagged by Dice < 0.45.\\n")
    f.write("This is a reproducible coursework simulation, not validation on real photographed library forms.\\n")
