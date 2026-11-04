import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for p in query_images()[:10]:
    im=read(p); g=gray(im); e=cv2.Canny(g,60,150)
    contours,_=cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); contour_img=im.copy()
    if contours:
        c=max(contours,key=cv2.contourArea); peri=cv2.arcLength(c,True); poly=cv2.approxPolyDP(c,.03*peri,True); cv2.polylines(contour_img,[poly],True,(0,255,0),3)
    hough=cv2.HoughLines(e,1,np.pi/180,100); std=im.copy()
    if hough is not None:
        for line in hough[:20]:
            rho,theta=line[0]; a,b=np.cos(theta),np.sin(theta); x0,y0=a*rho,b*rho
            cv2.line(std,(int(x0+1000*-b),int(y0+1000*a)),(int(x0-1000*-b),int(y0-1000*a)),(0,0,255),1)
    hp=cv2.HoughLinesP(e,1,np.pi/180,40,minLineLength=30,maxLineGap=10); prob=im.copy()
    if hp is not None:
        for x1,y1,x2,y2 in hp[:,0]: cv2.line(prob,(x1,y1),(x2,y2),(255,0,0),2)
    rows += [{"image":p.name,"method":"Contour approximation","detections":len(contours)},{"image":p.name,"method":"Standard Hough","detections":0 if hough is None else len(hough)},{"image":p.name,"method":"Probabilistic Hough","detections":0 if hp is None else len(hp)}]
    if p.name=="query_01.png": ims=[contour_img,std,prob]
save_board(6,ims,["Contours","Hough lines","Hough segments"])
pd.DataFrame(rows).to_csv(out(6,"hough_comparison.csv"),index=False)
