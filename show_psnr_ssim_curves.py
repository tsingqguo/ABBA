#!/usr/bin/env python
# coding: utf-8
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

warnings.filterwarnings("ignore")

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window

def check_image(image):
    # image should a 64 x 64 x 3 RGB image
    assert(isinstance(image, np.ndarray))

    if len(image.shape)>3:
        image = image.squeeze(0).transpose(1,2,0)

    assert(image.shape == (64, 64, 3) or image.shape == (224, 224, 3) or image.shape == (299, 299, 3))

    if image.dtype == np.float32:
        # we accept float32, but only if the values
        # are between 0 and 255 and we convert them
        # to integers
        if image.min() < 0:
            logger.warning('clipped value smaller than 0 to 0')
        if image.max() > 255:
            logger.warning('clipped value greater than 255 to 255')
        if image.max() <= 1:
            image = image*255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    assert image.dtype == np.uint8
    return image

def BRISQUE(img):
    img = Image.fromarray(np.uint8(img.squeeze(0).transpose(1,2,0)))
    bris_score = bris.score(img)
    return bris_score

def SSIM(img1, img2):

    if isinstance(img1,np.ndarray):
        img1 = torch.from_numpy(img1.astype(np.float32)).cuda()
    if isinstance(img2,np.ndarray):
        img2 = torch.from_numpy(img2.astype(np.float32)).cuda()

    (_, channel, _, _) = img1.size()
    # print(img1.size())
    window_size = 11
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=(np.uint8(window_size / 2), np.uint8(window_size / 2)), groups=channel)
    mu2 = F.conv2d(img2, window, padding=(np.uint8(window_size / 2), np.uint8(window_size / 2)), groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=(np.uint8(window_size / 2), np.uint8(window_size / 2)), groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=(np.uint8(window_size / 2), np.uint8(window_size / 2)), groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=(np.uint8(window_size / 2), np.uint8(window_size / 2)), groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def PSNR(img1, img2, _norm=False):
    img1 = img1.transpose(2, 3, 1, 0).squeeze(-1)
    img2 = img2.transpose(2, 3, 1, 0).squeeze(-1)

    img1 = check_image(img1)
    img2 = check_image(img2)

    if _norm:
        img1 = img1 / 255.
        img2 = img2 / 255.
        img1 = img1 / np.linalg.norm(img1)
        img2 = img2 / np.linalg.norm(img2)
        mse = np.mean((img1 - img2) ** 2)
    else:
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    #mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def PSNR_stru(img1, img2):
    if len(img1.shape)==4 and img1.shape[1]==3:
        img1 = img1.transpose(2, 3, 1, 0).squeeze(-1)
        img2 = img2.transpose(2, 3, 1, 0).squeeze(-1)

    img1 = check_image(img1)
    img2 = check_image(img2)

    mm=[]
    _mse = (img1 / 255. - img2 / 255.) ** 2
    _mse = np.mean(_mse, axis=2)
    for i in range(8):
        for j in range(8):
            mm.append(_mse[i:i+292,j:j+292])
    mm=np.array(mm) # 64 292 292
    mm=np.mean(mm,axis=0)
    PIXEL_MAX = 1
    eps = 2.22044604925031e-016
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mm)+eps)
    psnr = np.mean(psnr)
    if psnr > 100:
        psnr = 100
    #mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    #mse = np.mean((img1 - img2) ** 2)
    #if mse == 0:
    #    return 100
    #PIXEL_MAX = 1
    #return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

# ------- testing on dev dataset -----------------------#
def check_show_dev_res(method,eps_range,dataset="dev"):

    result_root = "/mnt/nvme/projects/BlurAttack/results/"
    dataset = dataset + "/"
    white_model_name = "inceptionv3"
    bbmodel_names = ["inceptionv3", "inceptionresnetv2", "inceptionv4", "xception"]
    slt_num = 1000
    valid_fold_list = []
    dataset_root="/mnt/nvme/dataset/dev/images/"

    # setup default parameters:
    if method == "fgsm":
        step_size= 1.0
        blur_strategy = ""
        method=method+"v0"
        file_ext_gt=".png"
    elif method == "mifgsm":
        step_size= 1.0
        blur_strategy = ""
        method = method + "v0"
        file_ext_gt=".png"
    elif method == "dim":
        step_size= 1.0
        blur_strategy = ""
        method=method+"v0"
        file_ext_gt=".png"
    elif method == "tifgsm":
        step_size= 1.0
        blur_strategy = ""
        method=method+"v0"
        file_ext_gt=".png"
    elif method == "timifgsm":
        step_size= 1.0
        blur_strategy = ""
        method = method + "v0"
        file_ext_gt=".png"
    elif method == "tidim":
        step_size= 1.0
        blur_strategy = ""
        method=method+"v0"
        file_ext_gt=".png"
    elif method == "gblur":
        step_size= -2.0
        blur_strategy = "_blur_strategy_whole"
        file_ext_gt=".jpg"
    elif method == "dblur":
        step_size= -2.0
        blur_strategy = "_blur_strategy_whole"
        file_ext_gt=".jpg"
    elif method == "mbAdv_mifgsm":
        step_size= -3.0
        blur_strategy = "_blur_strategy_joint"
        file_ext_gt=".jpg"

    method_name = white_model_name + "_" + white_model_name + "_" + method

    for eps in eps_range:
        if step_size == -1.0:
            step_size_ = round(eps/10,4)
        elif step_size == -2.0:
            step_size_ = eps
            eps = 1.0
        elif step_size == -3.0:
            step_size_ = 10
            eps1 = eps[1]
            eps2 = eps[0]
        else:
            step_size_ = step_size

        if step_size_ == 10:
            file_name = '/eplison_{}_{}'.format(eps1,eps2)+'_stepsize_{}'.format(step_size_)+blur_strategy
        else:
            eps_ = float(format(eps, '.4f'))
            if eps!=eps_:
                file_name = '/eplison_{}'.format(eps)+'_stepsize_{}'.format(step_size_)+blur_strategy
            elif eps==1.0:
                file_name = '/eplison_{}'.format(eps) + '_stepsize_{}'.format(step_size_) + blur_strategy
            else:
                file_name = '/eplison_{:0,.4f}'.format(eps)+'_stepsize_{}'.format(step_size_)+blur_strategy

        print(result_root+dataset+method_name+file_name)
        if not os.path.exists(result_root+dataset+method_name+file_name):
            assert(os.path.exists(result_root+dataset+method_name+file_name))
        valid_fold_list.append(result_root+dataset+method_name+file_name)

    ifpsnr,ifssim,ifbris = True,True,True
    avg_psnrs = []
    avg_ssims = []
    avg_succes = []
    avg_brises = []

    filenamelist = np.load("/home/guoqing/projects/BlurAttack/experiments/filenames_list.npy")

    for fold in valid_fold_list:
        print(fold)
        status_file_path = fold

        avg_psnrs_models = []
        avg_succes_models = []
        avg_ssims_models = []
        avg_bris_models = []

        for model_id, forward_model_name in enumerate(bbmodel_names):

            if forward_model_name == white_model_name and method not in ["mbAdv_mifgsm","gblur","dblur"]:
                res_path = status_file_path + "/{}_{}_succ_rate{}_.npy".format(white_model_name, forward_model_name, slt_num)
            else:
                res_path = status_file_path + "/{}_{}_succ_rate{}.npy".format(white_model_name, forward_model_name, slt_num)
            status = np.load(res_path)

            succ_ = np.zeros_like(status)
            fail_ = np.zeros_like(status)
            already_ = np.zeros_like(status)
            succ_[status==1.] = 1.
            fail_[status==-1.] = 1.
            already_[status==-0.] = 1.

            num_succ = succ_.sum()
            num_fail = fail_.sum()
            num_already = already_.sum()

            succ_rate = num_succ/(num_fail+num_already+num_succ)
            print("{}_bmodel:{}_fmodel:{}:success rate:{}".format(fold,white_model_name,forward_model_name,succ_rate))

            # find adversarial examples, remove fail results
            image_files = os.listdir(status_file_path)
            psnr = []
            ssim = []
            brisque = []

            if not os.path.exists(fold + "/psnr_ssim_bris_whitemodel.npz") and white_model_name == forward_model_name:

                for idx, file in enumerate(tqdm(filenamelist)):  # enumerate(tqdm(image_files)):
                    file = str(file[0][:])[2:-1]
                    full_path = status_file_path + "/" + file
                    full_path_withoutext, file_ext = os.path.splitext(full_path)
                    org_img_path = dataset_root + file
                    file_ext = file_ext_gt

                    if file_ext == file_ext_gt and full_path_withoutext[-4:] != "_org"  \
                            and method in ["fgsmv0","mifgsmv0","dimv0","tifgsmv0","timifgsmv0","tidimv0"]:

                        if status[idx] == 1.0:
                            # print("image:{}th:{}".format(idx, file))
                            img = imageio.imread(full_path)
                            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).numpy()
                            # img = np.load(full_path_withoutext+".npy")
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).numpy()

                            # calculate psnr and ssim
                            img_org = imageio.imread(org_img_path)
                            img_org = torch.from_numpy(img_org).permute(2, 0, 1).unsqueeze(0).numpy()
                            if (img_org - img).max() == 0:
                                continue
                            if ifpsnr:
                                psnr.append(PSNR(img, img_org))
                            else:
                                psnr.append(0.0)
                            if ifssim:
                                ssim.append(SSIM(img, img_org))
                            else:
                                ssim.append(0.0)
                            if ifbris:
                                brisque.append(BRISQUE(img))
                            else:
                                brisque.append(0.0)
                        else:
                            psnr.append(-1.0)
                            ssim.append(-1.0)
                            brisque.append(-1.0)

                    elif file_ext == file_ext_gt and full_path_withoutext[-4:]!="_org" \
                            and method in ["mbAdv_mifgsm","gblur","dblur"]:

                        #print("image:{}th:{}".format(idx, file))
                        if os.path.exists(full_path_withoutext+file_ext):
                            img = imageio.imread(full_path_withoutext+file_ext)
                            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).numpy()
                            #img = np.load(full_path_withoutext+".npy")
                            img_valid = (img.max() != 0)
                        else:
                            img_valid = False

                        if img_valid:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).numpy()
                            # calculate psnr and ssim
                            img_org = imageio.imread(org_img_path)
                            img_org = torch.from_numpy(img_org).permute(2,0,1).unsqueeze(0).numpy()
                            if (img_org-img).max() == 0:
                                continue
                            if ifpsnr:
                                psnr.append(PSNR(img,img_org))
                            else:
                                psnr.append(0.0)
                            if ifssim:
                                ssim.append(SSIM(img,img_org))
                            else:
                                ssim.append(0.0)
                            if ifbris:
                                brisque.append(BRISQUE(img))
                            else:
                                brisque.append(0.0)
                        else:

                            psnr.append(-1.0)
                            ssim.append(-1.0)
                            brisque.append(-1.0)

                # for idx,file in enumerate(tqdm(image_files)):
                #     full_path = status_file_path+"/"+file
                #     full_path_withoutext, file_ext = os.path.splitext(full_path)
                #     org_img_path = dataset_root+file
                #     if file_ext == file_ext_gt and full_path_withoutext[-4:]!="_org":
                #         #print("image:{}th:{}".format(idx, file))
                #         img = imageio.imread(full_path)
                #         img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).numpy()
                #         #img = np.load(full_path_withoutext+".npy")
                #         if img.max()<=1.0:
                #             img = (img*255).astype(np.uint8)
                #             img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).numpy()
                #
                #         if img.max()==0:
                #             #print("remove:{}".format(full_path))
                #             os.remove(full_path)
                #             os.remove(full_path_withoutext+"_org.jpg")
                #             if os.path.exists(full_path_withoutext + "_org.npy"):
                #                 os.remove(full_path_withoutext + "_org.npy")
                #         else:
                #             # calculate psnr and ssim
                #             img_org = imageio.imread(org_img_path)
                #             img_org = torch.from_numpy(img_org).permute(2,0,1).unsqueeze(0).numpy()
                #             if (img_org-img).max() == 0:
                #                 continue
                #             if ifpsnr:
                #                 psnr.append(PSNR(img,img_org))
                #             else:
                #                 psnr.append(0.0)
                #             if ifssim:
                #                 ssim.append(SSIM(img,img_org))
                #             else:
                #                 ssim.append(0.0)
                #             if ifbris:
                #                 brisque.append(BRISQUE(img))
                #             else:
                #                 brisque.append(0.0)

                psnrs = np.array(psnr)
                ssims = np.array(ssim)
                avg_succ = succ_rate
                brises = np.array(brisque)

                np.savez(fold + "/psnr_ssim_bris_whitemodel.npz", psnrs, avg_succ, ssims, brises)

            elif os.path.exists(fold + "/psnr_ssim_bris_whitemodel.npz") and white_model_name != forward_model_name:

                arrs = np.load(fold+"/psnr_ssim_bris_whitemodel.npz")
                psnrs, _, ssims, brises = arrs['arr_0'],arrs['arr_1'],arrs['arr_2'],arrs['arr_3']

                if white_model_name!=forward_model_name:
                    psnrs = psnrs[status==1]
                    ssims = ssims[status==1]
                    brises = brises[status==1]
                    avg_succ = succ_rate

            elif os.path.exists(fold + "/psnr_ssim_bris_whitemodel.npz") and white_model_name == forward_model_name:
                arrs = np.load(fold+"/psnr_ssim_bris_whitemodel.npz")
                psnrs, avg_succ, ssims, brises = arrs['arr_0'],arrs['arr_1'],arrs['arr_2'],arrs['arr_3']

            avg_psnrs_models.append(psnrs)
            avg_succes_models.append(avg_succ)
            avg_ssims_models.append(ssims)
            avg_bris_models.append(brises)

        np.savez(fold + "/psnr_ssim_bris_all.npz", avg_succes_models, avg_psnrs_models, avg_ssims_models, avg_bris_models)

        avg_psnrs.append(avg_psnrs_models)
        avg_ssims.append(avg_ssims_models)
        avg_succes.append(avg_succes_models)
        avg_brises.append(avg_bris_models)

    # pos-processing
    avg_psnrs_,avg_ssims_,avg_succes_,avg_brises_ = [],[],[],[]
    for avg_psnr,avg_ssim,avg_succ,avg_brise in zip(avg_psnrs,avg_ssims,avg_succes,avg_brises):
        avg_psnr_, avg_ssim_, avg_succ_, avg_brise_ = [], [], [], []
        for psnr,ssim,succ,brise in zip(avg_psnr,avg_ssim,avg_succ,avg_brise):
            psnr = np.array(psnr[psnr!=-1.]).mean()
            ssim = np.array(ssim[ssim!=-1]).mean()
            succ = succ
            brise = np.array(brise[brise!=-1]).mean()
            avg_psnr_.append(psnr)
            avg_ssim_.append(ssim)
            avg_succ_.append(succ)
            avg_brise_.append(brise)
        avg_psnrs_.append(avg_psnr_)
        avg_ssims_.append(avg_ssim_)
        avg_succes_.append(avg_succ_)
        avg_brises_.append(avg_brise_)

    avg_psnrs = np.array(avg_psnrs_)
    avg_ssims = np.array(avg_ssims_)
    avg_succes = np.array(avg_succes_)
    avg_brises = np.array(avg_brises_)


    return avg_succes,avg_psnrs,avg_ssims,avg_brises

def main(argv):

    opts, args = getopt.getopt(sys.argv[1:], "d:g:s:e:", ["dataset","gpu_id","startid","endid"])
    dataset = "dev"
    gpu_id=0
    start_id=0
    end_id = 9

    for op, value in opts:
        if op == '-d' or op == '--dataset':
            dataset = value
        if op == '-g' or op == '--gpu_id':
            gpu_id = value
        if op == '-s' or op == '--startid':
            start_id = int(value)
        if op == '-e' or op == '--endid':
            end_id = int(value)

    if dataset == 'dev':
        dataset_path = "/mnt/nvme/dataset/dev/images" #"/home/wangjian/tsingqguo/dataset/dev/images" #

    save_path = "/home/guoqing/projects/BlurAttack/experiments/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path+="psnr_ssim_succ_{}.npz"

    print('dataset path:{}'.format(dataset_path))
    methods = ["gblur","dblur","fgsm","mifgsm","dim","tifgsm","timifgsm","tidim","mbAdv_mifgsm"] #
    methods = methods[start_id:end_id]
    print(methods)
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)

    re_run = False

    avg_succes_methods= []
    avg_psnrs_methods = []
    avg_ssims_methods = []
    avg_brises_methods= []
    for method in methods:
        # if method == "fgsm":
        #     eps_range = np.round(np.arange(0.1,0.6,0.05),3).tolist()
        # elif method == "mifgsm":
        #     eps_range = np.round(np.arange(0.005,0.055,0.005),3).tolist()
        if method == "gblur":
            eps_range = [.0013, .0023, .0033, .0043, .0053, .0063, .0073, .0083, .0093,.0103]
        elif method == "dblur":
            eps_range = [3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0,21.0]
        elif method == "mbAdv_mifgsm":
            eps_range = [[5.0,0.1],[10.0,0.2],[15.0,0.3],[20.0,0.4],[25.0,0.5],\
                              [30.0,0.6],[35.0,0.7],[40.0,0.8],[45.0,0.9],[50.0,1.0]]
        elif method == "dim":
            eps_range = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.09375,0.1250,0.15625,0.1875,0.21875]
        elif method == "fgsm":
            eps_range = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.09375,0.1250,0.15625,0.1875,0.21875]
        elif method == "mifgsm":
            eps_range = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.09375,0.1250,0.15625,0.1875,0.21875]
        elif method == "tidim":
            eps_range = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.09375,0.1250,0.15625,0.1875,0.21875]
        elif method == "tifgsm":
            eps_range = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.09375,0.1250,0.15625,0.1875,0.21875]
        elif method == "timifgsm":
            eps_range = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.09375,0.1250,0.15625,0.1875,0.21875]

        if not os.path.exists(save_path.format(method)) or re_run:
            avg_succes_models, avg_psnrs_models,avg_ssims_models,avg_brises_models = check_show_dev_res(method,eps_range,dataset)
            np.savez(save_path.format(method),avg_succes_models,avg_psnrs_models,avg_ssims_models,avg_brises_models)
        else:
            arrs = np.load(save_path.format(method))
            avg_succes_models, avg_psnrs_models, avg_ssims_models,avg_brises_models = arrs['arr_0'],arrs['arr_1'],arrs['arr_2'],arrs['arr_3']
            # avg_succes_models, avg_psnrs_models, avg_ssims_models, avg_brises_models  = \
            #     avg_psnrs_models[0:10:2,:], avg_psnrs_models[0:10:2,:], avg_ssims_models[0:10:2,:], avg_brises_models[0:10:2,:]

        avg_succes_methods.append(avg_succes_models)
        avg_psnrs_methods.append(avg_psnrs_models)
        avg_ssims_methods.append(avg_ssims_models)
        avg_brises_methods.append(avg_brises_models)


    avg_succes_methods = np.array(avg_succes_methods)
    avg_psnrs_methods = np.array(avg_psnrs_methods)
    avg_ssims_methods = np.array(avg_ssims_methods)
    avg_brises_methods = np.array(avg_brises_methods)

    # visualization
    viz = visdom.Visdom()
    marker = {'color': 'red', 'symbol': 104, 'size': "10"}
    legend = ["GaussBlur","DefocBlur","FGSM","MIFGSM","DIM","TIFGSM","TIMIFGSM","TIDIM","ABBA"]
    white_model_name = "inceptionv3"
    bbmodel_names = ["inceptionv3", "inceptionresnetv2", "inceptionv4", "xception"]
    bbmodel_legends = ["Inc-v3", "IncRes-v2", "Inc-v4", "Xception"]

    # for vi in range(avg_succes_methods.shape[2]):
    #     avg_succes_methods_sub = avg_succes_methods[:,:,vi]
    #     avg_brises_methods_sub = avg_brises_methods[:, :, vi]
    #     viz.line(
    #         Y=avg_succes_methods_sub.transpose(1,0),
    #         X=avg_brises_methods_sub.transpose(1,0),#np.linspace(0, 9, 10)[:,np.newaxis].repeat(5,1),
    #         opts=dict(
    #             fillarea=False,
    #             showlegend=True,
    #             width=800,
    #             height=800,
    #             xlabel='BRISQUE',
    #             ylabel='Succ.Rate',
    #             #ytype='log',
    #             title='Succ.Rate vs. BRISQUE ({})'.format(bbmodel_names[vi]),
    #             marginleft=20,
    #             marginright=20,
    #             marginbottom=30,
    #             margintop=30,
    #             mode="markers+lines",
    #             legend=legend,
    #             marker=marker
    #         ),
    #         win="win{}".format(vi),
    #     )


    for vi in range(avg_succes_methods.shape[2]):
        fig = plt.figure()
        ax = fig.add_axes([0.09, 0.1, 0.85, 0.85])
        avg_succes_methods_sub = avg_succes_methods[:,:,vi]
        avg_brises_methods_sub = avg_brises_methods[:, :, vi]
        for i in range(avg_succes_methods_sub.shape[0]):
            avg_succes = avg_succes_methods_sub[i,:]
            avg_bris = avg_brises_methods_sub[i,:]
            ax.plot(avg_bris,avg_succes, marker='o', label=legend[i])
            if vi==0:
                ax.legend(loc = 'lower right', borderaxespad=0.5,prop={'size':15})
        minorLocatorx = MultipleLocator(7.0)
        minorLocatory = MultipleLocator(0.06)
        ax.yaxis.set_minor_locator(minorLocatory)
        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.set_xlabel('BRISQUE',fontsize=15)
        ax.set_ylabel('Succ.Rate',fontsize=15)
        ax.grid(True,ls='--',which = 'minor')
        if vi>0:
            plt.title('Succ.Rate vs. BRISQUE (Transfer to {})'.format(bbmodel_legends[vi]),fontsize=15)
        else:
            plt.title('Succ.Rate vs. BRISQUE (Attack {})'.format(bbmodel_legends[vi]),fontsize=15)
        plt.savefig('testeps{}.eps'.format(vi),format='eps',dpi=1000)
        viz.matplot(plt, win="matplot{}".format(vi))

    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
