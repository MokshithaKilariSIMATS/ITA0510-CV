import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
db=db_images(); qs=query_images(); labels=pd.read_csv(DATA/"query_labels.csv")
rows=[]; sample=[]
for idx,q in enumerate(qs):
    im=read(q); t=time.perf_counter(); enhanced=cv2.equalizeHist(gray(im)); low=(time.perf_counter()-t)*1000
    t=time.perf_counter(); k,d=orb_desc(im); mid=(time.perf_counter()-t)*1000
    # nearest product by ORB match score
    scores=[]
    for p in db:
        scores.append((match_score(d,orb_desc(read(p))[1]),p.name))
    high=(time.perf_counter()-t)*1000
    best=max(scores)[1] if scores else "none"
    pid=int(labels.iloc[idx].product_id); true_files=[p.name for p in db if f"_p{pid:02d}." in p.name]
    predicted=best.split("_p")[-1].split(".")[0] if "_p" in best else "-1"
    rows.append({"query":q.name,"ground_truth_product":pid,"predicted_product":predicted,"low_level_ms":low,"feature_search_ms":mid,"high_level_search_ms":high,"correct":str(pid)==predicted})
    if idx==0: sample=[im,enhanced]
pd.DataFrame(rows).to_csv(out(1,"solution_comparison.csv"),index=False)
save_board(1,sample,["Input packet","Low-level enhancement"])
