import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
db=db_images(); rows=[]; preview=[]
# Compare raw grayscale vector, HSV histogram and ORB local features on query set.
for q in query_images()[:20]:
    im=read(q); qv=resize_gray(im).flatten().astype(np.float32)
    qh=cv2.calcHist([im],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]); cv2.normalize(qh,qh)
    qk,qd=orb_desc(im); candidates=[]
    for p in db:
        ref=read(p); rv=resize_gray(ref).flatten().astype(np.float32)
        rh=cv2.calcHist([ref],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]); cv2.normalize(rh,rh)
        rk,rd=orb_desc(ref)
        pixel=float(np.linalg.norm(qv-rv)); hist=float(cv2.compareHist(qh,rh,cv2.HISTCMP_CORREL)); local=match_score(qd,rd)
        candidates.append((pixel,hist,local,p.name))
    bestpix=min(candidates,key=lambda x:x[0])[3]; besthist=max(candidates,key=lambda x:x[1])[3]; bestlocal=max(candidates,key=lambda x:x[2])[3]
    rows.append({"query":q.name,"pixel_top1":bestpix,"histogram_top1":besthist,"local_ORB_top1":bestlocal})
    if q.name=="query_01.png": preview=[im,read(db[0])]
save_board(9,preview,["Query packet","Example database packet"])
pd.DataFrame(rows).to_csv(out(9,"integrated_pipeline_results.csv"),index=False)
(ROOT/"Task_09"/"Output"/"system_block_diagram.txt").write_text(
"Camera -> Preprocessing -> Edge detection -> Packet outline / perspective correction -> Keypoints + descriptors -> PCA compression / global shortlist -> Local matching + verification -> Product ID -> Product name and price -> Bill\\n",encoding="utf-8")
