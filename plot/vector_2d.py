
import matplotlib.pyplot as plt
def vector(x,y):
    """
    x，y为相同长度的向量
    
    """
    if len(x) != len(y):
        raise Exception("xy向量长度必须相等")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
