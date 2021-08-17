import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib.backends.backend_pdf import PdfPages
HEIGHT_ALL, WIDTH_ALL = 30, 30
TARGETX, TARGETY = 15, 15
HEIGHT, WIDTH = 24, 24
TARGETX_P, TARGETY_P = 12, 12
dir1 = "./result_txt/v30_base"
dir2 = "./result_txt/v30_sparse"
store_type = "PDF"
# sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
for t in range(0,120):
    # if t % 10 ==0:
    print(t)
    virtual = dir1 + '/virtual/' + str(t) + '.txt'
    virtual_path = np.loadtxt(virtual)

    Ours = dir1 + '/Ours/' + str(t) + '.txt'
    Ours_path = np.loadtxt(Ours).reshape((-1,2))

    cmp = dir2 + '/Ours/' + str(t) + '.txt'
    cmp_path = np.loadtxt(cmp).reshape((-1,2))


    plt.axis("scaled")
    plt.xlim(0.0, WIDTH_ALL)
    plt.ylim(0.0, HEIGHT_ALL)
    plt.yticks([])
    plt.xticks([])


    if not os.path.exists(dir1 + '/virtual/pics'):
        os.mkdir(dir1 + '/virtual/pics')
    if store_type == 'PNG':
        dst_virtual = dir1 + '/virtual/pics/' + str(t) + '.png'
    else:
        dst_virtual = dir1 + '/virtual/pics/' + str(t) + '.pdf'
    x_virtual = virtual_path[:, 0]
    y_virtual = virtual_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
        # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t for t in range(len(x_virtual))]
    plt.scatter(np.array(x_virtual), np.array(y_virtual), s=0.5, c= c_l, alpha=0.2, cmap='Reds', label='virtual path')
    plt.scatter(TARGETX, TARGETY, c='gold', s=100, marker="*", label='target object', edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)


    # plt.legend()
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    if store_type == 'PNG':
        plt.savefig(dst_virtual)
        plt.clf()
    else:
        pp = PdfPages(dst_virtual)
        pp.savefig()
        pp.close()
        plt.close()

    plt.axis("scaled")
    plt.xlim(0.0, WIDTH)
    plt.ylim(0.0, HEIGHT)
    plt.yticks([])
    plt.xticks([])
    if not os.path.exists(dir1 + '/nor1'):
        os.mkdir(dir1 + '/nor1')
        # os.mkdir(dir1 + '/sparse/pics')
    if store_type == 'PNG':
        dst_cmp = dir1 + '/nor1/' + str(t) + '.png'
    else:
        dst_cmp = dir1 + '/nor1/' + str(t) + '.pdf'
    xv = cmp_path[:, 0]
    yv = cmp_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
        # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t*0.8  for t in range(len(xv))]
    plt.scatter(np.array(xv), np.array(yv), s=0.5, c= c_l, alpha=0.2, cmap='Blues')
    plt.scatter(xv[-1], yv[-1], c='b', alpha=0.2, label='cmp', s=0.5)
    x = Ours_path[:, 0]
    y = Ours_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
    # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t * 0.8 for t in range(len(x))]
    plt.scatter(np.array(x), np.array(y), s=0.5, c=c_l, alpha=0.2, cmap='Wistia')
    plt.scatter(x[-1], y[-1], c='g', alpha=0.2, label='base', s=0.5)

    # plt.legend(markerscale=50)
    plt.scatter(TARGETX_P, TARGETY_P, c='gold', s=100, marker="*", edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)
    # plt.legend(markerscale=10)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    if store_type =='PNG':

        plt.savefig(dst_cmp)
        plt.clf()
        plt.close()
    else:
        pp = PdfPages(dst_cmp)
        pp.savefig()
        pp.close()
        plt.close()