import os
import numpy as np
file = np.load('Dataset/eval_path_30_third.npy', allow_pickle=True)
if not os.path.exists('V30_third'):
    os.mkdir('V_30_third')
for i in range(file.shape[0]):
    if i % 100 == 0:
        print(i)
# for i in range(10):
    t = file[i]
    t = t[:,:3]
    np.savetxt('V_30_third/'+str(i)+'.txt', t)
# np.savetxt('lyjPathFile/v30.txt', out)
# print('end')
