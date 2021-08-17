import numpy as np
dir = '../Dataset/train_path_30'
for i in range(1,4):
    print(i)
    file = dir+"_" + str(i) + '.npy'
    if i == 1 :
        init_file = np.load(file, allow_pickle=True)
    else:
        c_file = np.load(file, allow_pickle=True)
        init_file = np.concatenate((init_file, c_file))

np.save(dir, init_file)