import torch
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

if __name__=="__main__":
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.2, 0.7]])
    print(y_hat[0][0],y_hat[1][2])
    print(cross_entropy(y_hat, y))
    