
import math
import numpy as np
from typing import Union
def normal(x: Union[float, list, np.ndarray], mu:int, sigma:int)->Union[float, list, np.ndarray]:
    """
    根据公式 写的正态分布函数

    mu: mean
    sigma: std 必须为正实数

    """
    if sigma <= 0:
        raise Exception("sigma必须为正实数")

    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

