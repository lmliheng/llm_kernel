import torch
def relu(X):
    """
    X是tensor张量
    """
    return torch.max(X,torch.zeros_like(X))

if __name__=='__main__':
    print(relu(torch.arange(-8,8,1)))
    print(relu(torch.arange(-8,8,1).reshape([32,2])))