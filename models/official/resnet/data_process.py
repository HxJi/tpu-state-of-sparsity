# import pandas as pd

# csv_data = np.loadtxt('sparsity-mask-ckpt.csv')
# print(csv_data.shape) 
# tmp1 = csv_data.reshape(53,90)
# print(tmp1.shape)

# np.savetxt('sparsity-mask-processed.csv', tmp1)


import pandas as pd
import numpy as np


with open('sparsity-mask-ckpt.csv') as f:
    lines = (line for line in f)
    FH = np.loadtxt(lines, delimiter=',', skiprows=1)

FHF = np.insert(FH,0,0,0)
tmp1 = FHF.reshape(91,53)
tmp2 = np.transpose(tmp1)
print(tmp2)
print(tmp2.shape)

np.savetxt('sparsity-mask-processed.csv', tmp2, delimiter = ',')