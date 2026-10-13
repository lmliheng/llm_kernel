import torch
def cross_entropy_loss(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


if __name__=="__main__":
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.2, 0.7]])
    print(y_hat[0][0],y_hat[1][2])
    print(cross_entropy_loss(y_hat, y))
    