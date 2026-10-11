import torch
def accuracy(y_hat, y):  #@save
    """
    计算分类精度

    分类精度即正确预测数量与总预测数量之比
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)

    cmp = y_hat.type(y.dtype) == y 
    return float(cmp.type(y.dtype).sum())/len(y)

if __name__=="__main__":
   
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.2, 0.7]])
    print(accuracy(y_hat,y))
