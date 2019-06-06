import numpy as np
from common.funcs import softmax, softmax2


a = [[1, 2, 3, 4, 5],
     [3, 5, 6, 8, 9],
     [0, 7, 2, 4, 6]]
a = np.array(a)
print(softmax(a) == softmax2(a))
