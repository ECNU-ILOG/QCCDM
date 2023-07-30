import numpy as np

a = np.array([[1, 2, 1], [0, 0, 1]])
c, d = np.where(a != 0)
for a, b in zip(c.tolist(), d.tolist()):
    print(a, b)