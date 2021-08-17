import numpy as np
import os
import matplotlib.pyplot as plt
vrst_dir = '/home/czy/Desktop/plotPath/Results5-23/Results/RDW_PH_Build_V_30_third/Sampled Metrics'
print("hello")
for t in range(120):
    file = vrst_dir + "/trialId_" + str(t) + "/userId_0" + '/user_real_positions.csv'
    path = np.loadtxt(file, delimiter=',')
    path = path + 12
    if not os.path.exists('/home/czy/MyWorkSpace/advanced_srl/result_txt/v30Vthird_final/vrst/'):
        os.mkdir('/home/czy/MyWorkSpace/advanced_srl/result_txt/v30Vthird_final/vrst/')
    np.savetxt('/home/czy/MyWorkSpace/advanced_srl/result_txt/v30Vthird_final/vrst/' + str(t) + '.txt', path)
    print(t)
