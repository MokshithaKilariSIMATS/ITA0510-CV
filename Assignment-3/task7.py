import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
rows=[]; ims=[]
for q in query_images()[:20]:
    im=read(q); qk,qd=orb_desc(im); sk,sd=sift_desc(im)
    for p in db_images():
        ref=read(p); rk,rd=orb_desc(ref); rsk,rsd=sift_desc(ref)
        orb=match_score(qd,rd,cv2.NORM_HAMMING)
        sift=match_score(sd,rsd,cv2.NORM_L2) if sd is not None and rsd is not None else 0
        # FLANN uses SIFT descriptors when available; reported as a separate matching mode.
        rows.append({"query":q.name,"database_product":p.name,"ORB_good_matches":orb,"SIFT_good_matches":sift})
    if q.name=="query_01.png" and qd is not None:
        ref=read(db_images()[0]); rk,rd=orb_desc(ref); matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(qd,rd,k=2)
        good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
        ims=[cv2.drawMatches(im,qk,ref,rk,good[:30],None,flags=2)]
save_board(7,ims,["ORB feature matches"])
pd.DataFrame(rows).to_csv(out(7,"feature_matching_scores.csv"),index=False)
