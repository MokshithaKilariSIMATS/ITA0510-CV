import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
db=db_images(); X=np.array([resize_gray(read(p)).flatten() for p in db],dtype=np.float32)
mean,eig=cv2.PCACompute(X,mean=None,maxComponents=min(64,len(X)))
proj=cv2.PCAProject(X,mean,eig)
rows=[]
for k in [8,16,32,64]:
    kk=min(k,proj.shape[1]); rows.append({"components":kk,"bytes_for_30_products":int(proj[:,:kk].nbytes),"bytes_for_500_products_est":int(proj[:,:kk].nbytes/len(db)*500),"variance_retained_est":float(np.sum(eig[:kk]**2)/np.sum(eig**2))})
pd.DataFrame(rows).to_csv(out(8,"pca_memory_variance.csv"),index=False)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot([r["components"] for r in rows],[r["variance_retained_est"] for r in rows],marker="o"); ax.set_xlabel("Components"); ax.set_ylabel("Cumulative variance (estimated)"); ax.grid(True); fig.tight_layout(); fig.savefig(out(8,"pca_variance_plot.png"),dpi=150); plt.close(fig)
# First eigen-packet components
pcs=[]
for i in range(min(3,eig.shape[0])):
    pcs.append(cv2.normalize(eig[i].reshape(160,128),None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
save_board(8,pcs,["PC1","PC2","PC3"])
