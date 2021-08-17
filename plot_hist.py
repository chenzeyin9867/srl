import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
file = 'result30_pthird_hist'
# file = 'hist_resultv30_test_hist.txt'
gain = np.loadtxt(file)
gt = gain[:, 0]
gr = gain[:, 1]
gc = gain[:, 2]
fig = plt.figure(figsize=(15,5))

# plt.hist(gt, bins=50, rwidth=2, color='royalblue')

plt.hist(gt, bins=50, rwidth=2, color='crimson', density=True)
# plt.show()
# plt.cla()
# plt.hist(gr-1, bins= 30,histtype='step')
# plt.hist(gr-1, bins= 30, stacked=True)
# plt.yscale('log')
# plt.show()
# plt.cla()
# plt.hist(gc, bins=30,  histtype='step')
# plt.yscale('log')


# ytick = np.arange([0, 1, 10, 100])
plt.yticks([0, 1, 10, 100])
plt.yscale('log')
plt.show()
plt.cla()
# plt.savefig()
# pp = PdfPages('center_density.pdf')
# pp.savefig()
# pp.close()