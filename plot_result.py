import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib.backends.backend_pdf import PdfPages
HEIGHT_ALL, WIDTH_ALL = 30, 30
TARGETX, TARGETY = 15, 15
HEIGHT, WIDTH = 24, 24
TARGETX_P, TARGETY_P = 8, 8
dir = "./result_txt/v30_discuss"
# sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
for t in range(120):
    # if t % 10 ==0:
    print(t)
    virtual = dir + '/virtual/' + str(t) + '.txt'
    virtual_path = np.loadtxt(virtual)

    none = dir + '/none/' + str(t) + '.txt'
    none_path = np.loadtxt(none)
    Ours = dir + '/Ours/' + str(t) + '.txt'
    Ours_path = np.loadtxt(Ours)
    frank = dir + '/frank/' + str(t) + '.txt'
    frank_path = np.loadtxt(frank)
    vrst = dir + '/vrst/' + str(t) + '.txt'
    vrst_path = np.loadtxt(vrst)


    plt.axis("scaled")
    plt.xlim(0.0, WIDTH_ALL)
    plt.ylim(0.0, HEIGHT_ALL)
    plt.yticks([])
    plt.xticks([])


    if not os.path.exists(dir + '/virtual/pics'):
        os.mkdir(dir + '/virtual/pics')
    dst_virtual = dir + '/virtual/pics/' + str(t) + '.pdf'
    x_virtual = virtual_path[:, 0]
    y_virtual = virtual_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
        # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t for t in range(len(x_virtual))]
    plt.scatter(np.array(x_virtual), np.array(y_virtual), s=0.5, c= c_l, alpha=0.2, cmap='Reds')
    plt.scatter(TARGETX, TARGETY, c='gold', s=100, marker="*" , edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)


    # plt.legend()
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)  ####设置上部坐标轴的粗细

    # plt.savefig(dst_virtual)
    # plt.clf()

    pp = PdfPages(dst_virtual)
    pp.savefig()
    pp.close()
    plt.close()

    plt.axis("scaled")
    plt.xlim(0.0, WIDTH)
    plt.ylim(0.0, HEIGHT)
    plt.yticks([])
    plt.xticks([])
    if not os.path.exists(dir + '/none/pics'):
        os.mkdir(dir + '/none/pics')
    dst_none = dir + '/none/pics/' + str(t) + '.pdf'
    xv = none_path[:, 0]
    yv = none_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
        # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t*0.8  for t in range(len(xv))]
    plt.scatter(np.array(xv), np.array(yv), s=0.5, c= c_l, alpha=0.2, cmap='Blues')
    plt.scatter(TARGETX_P, TARGETY_P, c='gold', s=100, marker="*", label='target object', edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)

    # plt.legend()
    ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)  ####设置上部坐标轴的粗细
    # plt.savefig(dst_none)
    # plt.clf()
    # plt.close()

    pp = PdfPages(dst_none)
    pp.savefig()
    pp.close()
    plt.close()

    # Ours
    plt.axis("scaled")
    plt.xlim(0.0, WIDTH)
    plt.ylim(0.0, HEIGHT)
    plt.yticks([])
    plt.xticks([])
    if not os.path.exists(dir + '/Ours/pics'):
        os.mkdir(dir + '/Ours/pics')
    dst_ours = dir + '/Ours/pics/' + str(t) + '.pdf'
    x = Ours_path[:, 0]
    y = Ours_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
    # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t * 0.8 for t in range(len(x))]
    plt.scatter(np.array(x), np.array(y), s=0.5, c=c_l, alpha=0.2, cmap='Wistia')
    plt.scatter(TARGETX_P, TARGETY_P, c='gold', s=100, marker="*", label='target object', edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)

    plt.legend()
    # plt.savefig(dst_ours)
    # plt.clf()
    # plt.close()
    pp = PdfPages(dst_ours)
    pp.savefig()
    pp.close()
    plt.close()

    #frank
    # Ours
    plt.axis("scaled")
    plt.xlim(0.0, WIDTH)
    plt.ylim(0.0, HEIGHT)
    plt.yticks([])
    plt.xticks([])
    if not os.path.exists(dir + '/frank/pics'):
        os.mkdir(dir + '/frank/pics')
    dst_frank = dir + '/frank/pics/' + str(t) + '.pdf'
    x_f = frank_path[:, 0]
    y_f = frank_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
    # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t * 0.8 for t in range(len(x_f))]
    plt.scatter(np.array(x_f), np.array(y_f), s=0.5, c=c_l, alpha=0.2, cmap='Purples')
    plt.scatter(TARGETX_P, TARGETY_P, c='gold', s=100, marker="*", label='target object', edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)

    plt.legend()
    # plt.savefig(dst_frank)
    # plt.clf()
    # plt.close()
    ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)  ####设置上部坐标轴的粗细
    pp = PdfPages(dst_frank)
    pp.savefig()
    pp.close()
    plt.close()
    # vrst
    plt.axis("scaled")
    plt.xlim(0.0, WIDTH)
    plt.ylim(0.0, HEIGHT)
    plt.yticks([])
    plt.xticks([])
    if not os.path.exists(dir + '/vrst/pics'):
        os.mkdir(dir + '/vrst/pics')
    dst_vrst = dir + '/vrst/pics/' + str(t) + '.pdf'
    x_v = vrst_path[:, 0]
    y_v = vrst_path[:, 1]
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
    # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t * 0.8 for t in range(len(x_v))]
    plt.scatter(np.array(x_v), np.array(y_v), s=0.5, c=c_l, alpha=0.2, cmap='Greys')
    plt.scatter(TARGETX_P, TARGETY_P, c='gold', s=100, marker="*", label='target object', edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)

    plt.legend()
    ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)  ####设置上部坐标轴的粗细
    #
    # plt.savefig(dst_vrst)
    # plt.clf()
    # plt.close()
    pp = PdfPages(dst_vrst)
    pp.savefig()
    pp.close()
    plt.close()



    # all
    plt.axis("scaled")
    plt.xlim(0.0, WIDTH)
    plt.ylim(0.0, HEIGHT)
    plt.yticks([])
    plt.xticks([])
    if not os.path.exists(dir + '/all'):
        os.mkdir(dir + '/all')
    if not os.path.exists(dir + '/all/pics'):
        os.mkdir(dir + '/all/pics')
    dst_all = dir + '/all/pics/' + str(t) + '.pdf'
    #
    # for i in range(len(xv)):
    #             #     # print(float(i/len(x)))
    #     plt.scatter(np.array(xv[i]), np.array(yv[i]), s=0.5, c='r',
    #                             alpha=1.0 * math.exp(4 * (i / len(xv) - 1.0)))
    # plt.scatter(Xt, Yt, c='r', s=1)

    c_l = [t*0.8  for t in range(len(xv))]
    plt.scatter(np.array(xv), np.array(yv), s=0.5, c= c_l, alpha=0.2, cmap='Blues')
    # plt.scatter(xv[-1], yv[-1], c='b', label='NONE')
    c_l = [t * 0.8 for t in range(len(x))]
    plt.scatter(np.array(x), np.array(y), s=0.5, c=c_l, alpha=0.2, cmap='Wistia')
    # plt.scatter(x[-1], y[-1], c='g', label='OURS')
    # c_l = [t * 0.8 for t in range(len(x_f))]
    # plt.scatter(np.array(x_f), np.array(y_f), s=0.5, c=c_l, alpha=0.2, cmap='Greens')
    # plt.scatter(x_f[-1], y_f[-1], c='purple', label='FRANK')
    # c_l = [t * 0.8 for t in range(len(x_v))]
    # plt.scatter(np.array(x_v), np.array(y_v), s=0.5, c=c_l, alpha=0.2, cmap='Purples')
    # plt.scatter(x_v[-1], y_v[-1], c='grey', label='VRST')
    plt.scatter(TARGETX_P, TARGETY_P, c='gold', s=100, marker="*", edgecolors='orange',
                linewidths=0.5, cmap=plt.cm.Spectral)

    # plt.legend()
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)  ####设置上部坐标轴的粗细
    #
    # plt.savefig(dst_all)
    # plt.clf()
    # plt.close()
    pp = PdfPages(dst_all)
    pp.savefig()
    pp.close()
    # plt.close()