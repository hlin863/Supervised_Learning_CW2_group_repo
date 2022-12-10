import numpy as np
from sklearn.neighbors import KDTree

def load_dataset(name):
    path = "Datasets/{}".format(name)
    with open(path) as f:
        arr = []
        for line in f:
            line = line.strip()
            arr.append(line.split("\t"))
    D = np.array(arr, dtype=np.float32)
    Y = D[:,0].reshape(-1,1)
    y = (Y == 1) * 1 + (Y != 1) * -1
    return (D[:,1:], y)

def make_adj(X):
    W = np.zeros((X.shape[0], X.shape[0]))
    kd = KDTree(X)
    idx = kd.query(X, k=3, return_distance=False)
    for i in range(idx.shape[0]):
        for j in idx[i]:
            W[i,j] = (j != i) * 1
            W[j,i] = W[i,j]
    D = np.diag(np.sum(W, axis=1).T)
    return W, D, (D - W)

def harmonic(X, y, l):
    l_end = len(l)
    all_idx = np.arange(start=0, stop=y.shape[0], step=1, dtype=np.uint32)
    l_msk = np.ones(y.shape[0], dtype=bool)
    l_msk[l] = False
    u = all_idx[l_msk]
    X_ = np.vstack([X[l,:], X[u,:]])
    y_ = np.vstack([y[l,:], y[u,:]])
    W,_,L = make_adj(X_)
    L_inv = np.linalg.pinv(L[l_end:, l_end:])
    interploated = L_inv @ W[l_end:, :l_end] @ y_[:l_end]
    return y_, np.vstack([y_[:l_end], np.sign(interploated)])

def harmonic_kernel(X, y, l):
    _,_,L = make_adj(X)
    l_len = len(l)
    L_inv = np.linalg.pinv(L)
    K = np.zeros((l_len, l_len))
    for i in range(l_len):
        for j in range(l_len):
            K[i, j] = L_inv[l[i],l[j]]
    y_l = y[l]
    alpha = np.linalg.pinv(K) @ y_l
    ae = np.zeros((1, L.shape[0]))
    for i, idx in enumerate(l):
        ae[0, idx] = alpha[i]
    return y, np.sign(ae @ L_inv).T

def sample_labels(y, num):
    return np.random.choice(y, size=(num,))

def emp_err(gt, pred):
    return (gt != pred).astype(int).sum() / len(gt)

def laplacian_introp(X, y, sample_idx):
    sz = len(sample_idx)
    gt_all, pred_all = harmonic(X, y, sample_idx)
    return emp_err(gt_all[sz:], pred_all[sz:])

def laplacian_kernel_introp(X, y, sample_idx):
    gt_all, pred_all = harmonic_kernel(X, y, sample_idx)
    return emp_err(gt_all, pred_all)

def run_protocol():
    ds_size = [50, 100, 200, 400]
    l_size = [1,2,4,8,16]
    tables = np.zeros((len(ds_size), len(l_size), 2, 2))
    for i, sz in enumerate(ds_size):
        X, y = load_dataset("dtrain13_{}.dat".format(sz))
        clss_1 = np.arange(0, sz, 1, int)
        clss_2 = np.arange(sz, sz * 2, 1, int)
        for j, l in enumerate(l_size):
            errLI = []
            errKLI = []
            for _ in range(20):
                sampled_clss_1 = np.random.choice(clss_1, (l,), replace=False)
                sampled_clss_2 = np.random.choice(clss_2, (l,), replace=False)
                L_cal = np.hstack([sampled_clss_1, sampled_clss_2])
                errLI.append(laplacian_introp(X, y, L_cal))
                errKLI.append(laplacian_kernel_introp(X, y, L_cal))
            tables[i, j, 0, 0] = np.mean(errLI)
            tables[i, j, 1, 0] = np.std(errLI)
            tables[i, j, 0, 1] = np.mean(errKLI)
            tables[i, j, 1, 1] = np.std(errKLI)
            print("size={}, l={}: eli={}, ekli={}".format(sz, l, tables[i, j, :, 0], tables[i, j, :, 1]))
    return tables

def main():
    tables = run_protocol()
    print("\nSemi-norm (mean):")
    print(tables[:,:, 0, 0])
    print("\nKernelized (mean):")
    print(tables[:,:, 0, 1])

if __name__ == "__main__":
    main()