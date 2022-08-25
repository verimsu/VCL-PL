from tqdm import tqdm
import numpy as np
import faiss

X = np.load('X.npy')[:1627,:,:]
X_YFCC = np.load('X_YFCC.npy')
y = np.load('y.npy')[:1627]

y_YFCC_weighted = np.ones((X_YFCC.shape[0],40))

for c in tqdm(range(40)):
    xp = np.ascontiguousarray(X[:,c,:])
    
    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatL2(128)  # build a flat (CPU) index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.train(xp)
    gpu_index_flat.add(xp)    
    
    xq = np.ascontiguousarray(X_YFCC[:,c,:], dtype=np.float32)
    d, neighbors = gpu_index_flat.search(xq, 1)
    _y = np.multiply((2*y[:,c][neighbors]-1),np.exp(-d))
    _y = ((_y + 1)/2)
    y_YFCC_weighted[:,c] = np.squeeze(_y)
    
with open('y_YFCC_weighted_001.npy', 'wb') as f:
    np.save(f, y_YFCC_weighted)