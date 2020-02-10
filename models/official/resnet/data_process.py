# import pandas as pd

# csv_data = np.loadtxt('sparsity-mask-ckpt.csv')
# print(csv_data.shape) 
# tmp1 = csv_data.reshape(53,90)
# print(tmp1.shape)

# np.savetxt('sparsity-mask-processed.csv', tmp1)


import pandas as pd
import numpy as np
import os

# deal with mask sparsity
# with open('sparsity-mask-ckpt-102.log') as f:
#     lines = (line for line in f)
#     FH = np.loadtxt(lines, delimiter=',', skiprows=1)

# FHF = np.insert(FH,0,0,0)
# tmp1 = FHF.reshape(102,53)
# tmp2 = np.transpose(tmp1)
# print(tmp2)
# print(tmp2.shape)

# np.savetxt('sparsity-mask-processed-102.csv', tmp2, delimiter = ',')

#deal with activation sparsity
init = list(range(1,49))
df = pd.DataFrame(columns=['epoch'],data=init)
path = '/home/hj14/tpu/models/official/resnet/act-sparsity-logs/'
f_list = os.listdir(path)
for i in f_list:
    number = (i.split('-')[-1]).split('.')[0]
    epoch = str(int(number)/1251)
    with open(path+i) as f:
        lines = []
        for line in f:
            line = line[1:-2]
            if line.startswith('0.'):
                lines.append(line)
    df['epoch'+epoch]=lines
print(df)
df.to_csv(path+'sparsity-act-processed-102.csv')