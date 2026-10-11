# import numpy as np

# def softmax(logits:list|np.ndarray)->np.ndarray:
#     logits = np.asarray(logits, dtype=np.float64)


import torch
def softmax(X):
    """
    
    基于torch的softmax函数 和torch.nn.functional.softmax一样
    X是任意张量

    """
    X_max = X.max(dim=-1, keepdim=True).values
    X_stable = X - X_max
    
    X_exp = torch.exp(X_stable)
    partition = X_exp.sum(dim=-1, keepdim=True)
    return X_exp / partition

if __name__=="__main__":
    # print(softmax(torch.tensor([[1,2,3],[4,1,1]])))
    a=torch.randn([3,1,2])
    print(a)
    print(softmax(a))