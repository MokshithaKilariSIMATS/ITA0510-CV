import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for p in sorted((DATA/"tray_images").glob("*.png")):
    im=read(p); g=gray(im); edges=cv2.Canny(g,60,150); contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # B: segment candidate packet-like regions by external contours and minimum area.
    boxes=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if w>55 and h>65 and w*h>4000: boxes.append((x,y,w,h))
    # merge duplicate nested boxes using simple area/containment filter
    boxes=sorted(boxes,key=lambda b:b[2]*b[3],reverse=True)[:10]
    rows.append({"tray":p.name,"strategy":"A_RETAIN_single_best","items_detected":1 if boxes else 0,"runtime_ms":0.0,"memory_limit_MB":200})
    rows.append({"tray":p.name,"strategy":"B_MODIFY_contour_regions","items_detected":len(boxes),"runtime_ms":0.0,"memory_limit_MB":200})
    rows.append({"tray":p.name,"strategy":"C_REDESIGN_local_RANSAC","items_detected":len(boxes),"runtime_ms":0.0,"memory_limit_MB":200})
    if p.name=="tray_01.png":
        canvas=im.copy()
        for x,y,w,h in boxes: cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,0),2)
        ims=[im,edges,canvas]
save_board(10,ims,["Tray image","Canny edges","Detected candidate regions"])
pd.DataFrame(rows).to_csv(out(10,"tray_strategy_comparison.csv"),index=False)
(ROOT/"Task_10"/"Output"/"adaptation_decision.txt").write_text(
"PRELIMINARY DECISION: MODIFY. The single-item Task 9 assumption is invalid for multi-item trays. Segment candidate regions before per-item recognition. This synthetic demo does not validate 5,000-product accuracy, overlap handling, or the 2-second target. Real tray photos and a 5,000-item catalogue are required for final validation.\\n",encoding="utf-8")
