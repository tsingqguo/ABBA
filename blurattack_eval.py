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

warnings.filterwarnings("ignore")

# ------- testing on dev dataset -----------------------#
def check_show_dev_res(data_dir,prefix,white_model_name= "inceptionv3",slt_fold=None,dataset="dev",gpuid=0,recheck = False, extra_file=None, skip_eval = False):

    result_root = "./results/"
    check_file_path = result_root+dataset+"_valid_folds.npy"
    dataset = dataset + "/"
    prefix  = prefix + "/"
    bbmodel_names = ["inceptionresnetv2", "inceptionv3", "inceptionv4", "xception"]
    slt_num = 1000
    valid_fold_list = []
    generate_npy = False

    if slt_fold is not None:
        valid_fold_list.append(slt_fold)
    else:

        if os.path.exists(check_file_path) and not recheck:
            valid_fold_list = np.load(check_file_path).tolist()
        else:
            valdir = os.path.join(data_dir)
            batch_size = 1
            slt_num = 1000

            files = os.listdir(result_root+dataset+prefix)
            res_folds = []
            for file in tqdm(files):
                if os.path.isdir(result_root+dataset+prefix+file):
                    res_folds.append(file)
                    subfiles = os.listdir(result_root+dataset+prefix+file)
                    k=0
                    for subfile in subfiles:
                        if os.path.isdir(result_root+dataset+prefix+file+"/"+subfile):
                            print(subfile)
                            if k==0:
                                res_folds.pop()
                            res_folds.append(file+"/"+subfile)
                        k+=1

            if os.path.exists(check_file_path) and not recheck:
                valid_fold_list = np.load(check_file_path).tolist()
            else:
                valid_fold_list = []

            success_status = np.ones([slt_num]) * -1.
            success_status_fmodels = []
            fb_models = []

            for forward_model_name in bbmodel_names:
                success_status_fmodels.append(np.ones([slt_num]) * -1.)
                forward_model = create_fmodel("imagenet", model_name=forward_model_name, gpu=gpuid)
                fb_models.append(forward_model)

            for res_fold in tqdm(res_folds):

                if res_fold in valid_fold_list:
                    continue

                valid_fold_list.append(res_fold)

                # check if attack status files are existing:
                status_file_path = result_root + dataset + prefix+res_fold
                invalid_fold = True

                # if succ_rate doest not exist, we have to generate it
                if not os.path.exists(status_file_path + \
                    "/{}_{}_succ_rate{}.npy".format(white_model_name, white_model_name, slt_num)):
                    valid_fold_list.pop()

            np.save(check_file_path,valid_fold_list)


    for fold in valid_fold_list:
        print(fold)
        status_file_path = result_root + dataset + prefix + fold
        for forward_model_name in bbmodel_names:
            res_path = status_file_path + "/{}_{}_succ_rate{}.npy".format(white_model_name, forward_model_name, slt_num)
            status = np.load(res_path)

            if extra_file is not None:
                print(extra_file)
                is_advs = np.zeros_like(status)
                preds = np.zeros_like(status)
                res_path = status_file_path+"/"+extra_file
                _, res_ext = os.path.splitext(extra_file)

                if res_ext == ".txt":
                    import re
                    f = open(res_path)
                    line = f.readline()
                    k=0
                    imgnames = []
                    while line:
                        matchObj = re.match(r'(.*).png,(.*),[[](.*)[]]',line, re.M | re.I)
                        if matchObj:
                            imgnames.append(matchObj.group(1))
                            preds[k] = int(matchObj.group(2))-1
                            is_advs[k] = int(matchObj.group(3))
                        line = f.readline()
                        k+=1
                    f.close()
                    status_ = is_advs
                    status_[is_advs==0.] = -1

                    import pandas as pd
                    target_df = pd.read_csv(os.path.join(data_dir, 'dev_dataset.csv'), header=None)
                    f_to_true = dict(zip(target_df[0][1:].tolist(), [x - 1 for x in list(map(int, target_df[6][1:]))]))

                    for index_ in range(len(imgnames)):
                        imgname,_ = os.path.splitext(imgnames[index_])
                        true_label = f_to_true[imgname] if f_to_true[imgname] else 0
                        if is_advs[index_] == 1.:
                            if true_label == preds[index_]:
                                status_[index_]=-1
                            else:
                                status_[index_] = 1
                        #print("pred_label:{} true_label:{}".format(preds[index_],true_label))
                    status = status_

                elif res_ext==".npz":

                    data = np.load(res_path,allow_pickle=True)
                    pred_lbl = data["pred"]
                    true_lbl = data["truth"]
                    true_lbl2 = data["lbl_text"]
                    pred_logit = data["logit"]
                    _, uniq_idx = np.unique(true_lbl2, return_index=True)

                    pred_lbl = pred_lbl[uniq_idx]
                    true_lbl = true_lbl[uniq_idx]
                    #true_lbl2 = true_lbl2[uniq_idx]
                    #pred_logit = pred_logit[uniq_idx]
                    status = np.ones_like(pred_lbl)
                    status[pred_lbl==0]=-1
                    status[pred_lbl==true_lbl]=-1

            #print(status)

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
            for file in image_files:
                full_path = status_file_path+"/"+file
                full_path_withoutext, file_ext = os.path.splitext(full_path)
                if file_ext == ".jpg" and full_path_withoutext[-4:]!="_org":
                    img = imageio.imread(full_path)
                    if img.max()==0:
                        #print("remove:{}".format(full_path))
                        os.remove(full_path)
                        os.remove(full_path_withoutext+"_org.jpg")
                    if os.path.exists(full_path_withoutext + "_org.npy"):
                        os.remove(full_path_withoutext + "_org.npy")

    return valid_fold_list

# ------- testing on imagenet dataset -----------------------#
def check_show_imagenet_res(data_dir,dataset,gpuid=0,recheck = False):

    result_root = "/mnt/nvme/projects/BlurAttack/results/"
    check_file_path = result_root+dataset+"_valid_folds.npy"
    dataset = dataset + "/"
    bmodel_name = "resnet50"
    fmodel_names = ["resnet50", "densenet121", "pyramidnet101_a360"]
    slt_num = 1000

    if os.path.exists(check_file_path) and ~recheck:
        valid_fold_list = np.load(check_file_path).tolist()
    else:
        valdir = os.path.join(data_dir, 'val')
        slt_name = result_root+"/imagenet_slt_" + str(slt_num) + ".npy"
        sltIdx = np.load(slt_name)
        sltIdx.sort(axis=0)

        valid_sampler = torch.utils.data.SubsetRandomSampler(sltIdx.tolist())

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True, sampler=valid_sampler)

        files = os.listdir(result_root+dataset)
        res_folds = []
        for file in tqdm(files):
            if os.path.isdir(result_root+dataset+file):
                res_folds.append(file)
                subfiles = os.listdir(result_root+dataset+file)
                for subfile in subfiles:
                    if os.path.isdir(result_root+dataset+file+"/"+subfile):
                        res_folds.pop()
                        res_folds.append(file+"/"+subfile)

        if os.path.exists(check_file_path) and ~recheck:
            valid_fold_list = np.load(check_file_path).tolist()
        else:
            valid_fold_list = []

        fmodels = []
        for fmodel_name in fmodel_names:
            fmodel = create_fmodel("imagenet", model_name=fmodel_name, gpu=gpuid)
            fmodels.append(fmodel)

        for res_fold in tqdm(res_folds):
            if res_fold in valid_fold_list:
                continue
            valid_fold_list.append(res_fold)

            # check if attack status files are existing:
            status_file_path = result_root + dataset + res_fold
            invalid_fold = True
            k=0
            for fmodel in fmodels:
                if not os.path.exists(status_file_path+"/{}_{}_succ_rate{}.npy".format
                    (bmodel_name, fmodel_names[k], slt_num)):
                    success_status = np.ones([slt_num]) * -1.
                    # if succ_rate doest not exist, we have to generate it
                    for i, (images, labels, index, sample_path) in enumerate(tqdm(val_loader)):
                        file_path,file_full_name = os.path.split(sample_path[0])
                        file_name_, ext = os.path.splitext(file_full_name)
                        file_name_ = dataset+res_fold + "/"+file_name_
                        index = index.numpy()[0]

                        if os.path.exists(os.path.join(result_root,file_name_+".jpg")):
                            success_status[sltIdx==index], _, adversarial = load_adversarial(file_name_)
                            print(file_name_+" exists!")
                            # do blackbox attack
                            if success_status[sltIdx==index] == 1:
                                if adversarial.max() > 1:
                                    adversarial = adversarial.transpose(2, 0, 1) / 255
                                else:
                                    adversarial = adversarial.transpose(2, 0, 1)
                                adversarial = adversarial.astype("float32")
                                predictions = fmodel.forward_one(adversarial)
                                criterion1 = foolbox.criteria.Misclassification()
                                if criterion1.is_adversarial(predictions, labels):
                                    success_status[sltIdx==index] =1
                                else:
                                    success_status[sltIdx==index] = 0
                            continue
                        else:
                            invalid_fold = False
                            break
                    np.save(status_file_path + "/{}_{}_succ_rate{}.npy".format(bmodel_name, fmodel_names[k], slt_num),
                            success_status)
                k+=1

                if invalid_fold==False:
                    valid_fold_list.pop()
                    break

        np.save(check_file_path,valid_fold_list)

    for fold in valid_fold_list:
        status_file_path = result_root + dataset + fold
        for fmodel_name in fmodel_names:

            res_path = status_file_path + "/{}_{}_succ_rate{}.npy".format(bmodel_name, fmodel_name, slt_num)
            status = np.load(res_path)
            succ_ = np.zeros_like(status)
            fail_ = np.zeros_like(status)
            already_ = np.zeros_like(status)
            succ_[status==1.] = 1.
            fail_[status==0.] = 1.
            already_[status==-1.] = 1.

            num_succ = succ_.sum()
            num_fail = fail_.sum()
            num_already = already_.sum()

            succ_rate = num_succ/(num_fail+num_already+num_succ)
            print("{}_bmodel:{}_fmodel:{}:success rate:{}".format(fold,bmodel_name,fmodel_name,succ_rate))

            # print the image names


    return valid_fold_list

def main(argv):

    opts, args = getopt.getopt(sys.argv[1:], "d:g:p:r:s:e:w:", ["dataset","gpuid","prefix","recheck","slt_fold","extra_file","white_model_name"])
    dataset = "dev"
    gpuid = 0
    prefix = "inceptionv3_inceptionv3_mbAdv_mifgsm"
    recheck = 1
    slt_fold = None
    extra_file = None
    white_model_name = "inceptionv3"
    for op, value in opts:
        if op == '-d' or op == '--dataset':
            dataset = value
        if op == '-g' or op == '--gpuid':
            gpuid = value
        if op == '-p' or op == '--prefix':
            prefix = value
        if op == '-r' or op == '--recheck':
            recheck = int(value)
        if op == '-s' or op == '--slt_fold':
            slt_fold = value
        if op == '-e' or op == '--extra_file':
            extra_file = value
        if op == '-w' or op == '--white_model_name':
            white_model_name = value

    print("extra_file:{}".format(extra_file))
    if dataset == 'imagenet':
        dataset_path = "./dataset/ILSVRC2012"
    elif dataset == 'cifa10':
        dataset_path = "./dataset/cifar-10-batches-py"
    elif dataset == 'dev':
        dataset_path = "./dataset/dev/images" #"/home/wangjian/tsingqguo/dataset/dev/images" #
    elif dataset == 'minist':
        dataset_path = "./dataset/dev/minist"

    print('dataset path:{}'.format(dataset_path))
    print("prefix:{}".format(prefix))

    check_show_dev_res(dataset_path,prefix,white_model_name,slt_fold,dataset,gpuid,recheck,extra_file=extra_file)

    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
