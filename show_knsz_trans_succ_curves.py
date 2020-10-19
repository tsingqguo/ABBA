#!/usr/bin/env python
# coding: utf-8

# In[7]:
import os
import numpy as np
from utils import read_images, store_adversarial, load_adversarial, compute_MAD
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import warnings
import sys, getopt
from tqdm import tqdm
import foolbox
from fmodel import create_fmodel
import imageio
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math
import visdom
import imquality.brisque as bris
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc("pdf", fonttype=42)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
warnings.filterwarnings("ignore")

def main(argv):

    opts, args = getopt.getopt(sys.argv[1:], "d:g:s:e:", ["dataset","gpu_id"])
    dataset = "mnist"
    gpu_id=0

    for op, value in opts:
        if op == '-d' or op == '--dataset':
            dataset = value
        if op == '-g' or op == '--gpu_id':
            gpu_id = value

    if dataset == 'mnist':
        dataset_path = "./datasets/MNIST/"

    print('dataset path:{}'.format(dataset_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)

    save_path = "/mnt/nvme/projects/BlurAttack/results/mnist/stn_stn_mbAdv_mifgsm/"

    res_all = []
    for ts in np.arange(0.5,1.01,0.1):
        res_ts = []
        ts = round(ts, 1)
        for ks in np.arange(10.0,25.1,5.0):
            ks = round(ks, 2)
            res_path = "eplison_{}_{}_stepsize_20_blur_strategy_joint".format(ts,ks)
            res_path =save_path+"/"+res_path +"/stn_stn_succ_rate10000.npy"
            res = np.load(res_path)
            acc_org = (res != 0.0).sum()/len(res)
            acc_cur = (res == -1.0).sum()/len(res)
            succ_rate = (acc_org-acc_cur)/acc_org
            res_ts.append(succ_rate)
        res_all.append(res_ts)
    res_all = np.array(res_all)

    viz = visdom.Visdom()

    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0],projection='3d')
    _x = np.arange(0.5,1.01,0.1)
    _y = np.arange(10.0,25.1,5.0)
    _xx, _yy = np.meshgrid(_x, _y)

    _x_ind = np.arange(len(_x))
    _y_ind = np.arange(len(_y))
    _xx_ind, _yy_ind = np.meshgrid(_x_ind, _y_ind)

    x_ind, y_ind = _xx_ind.ravel(), _yy_ind.ravel()
    x, y = _xx.ravel(), _yy.ravel()

    top = res_all[x_ind,y_ind]
    bottom = np.zeros_like(top)
    width = 0.09
    depth = 4.5

    #
    norm = plt.Normalize(top.min(), top.max())
    norm_z = norm(top)
    map_vir = cm.get_cmap(name='cool')
    color = map_vir(norm_z)


    ax.bar3d(x, y, bottom, width, depth, top,color=color,shade=True,alpha=0.95)
    ax.set_title('Succ. Rate')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Succ. Rate')

    plt.savefig('stn_ks_ts.pdf',format='pdf',dpi=1000)
    viz.matplot(plt, win="matplot")


    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
