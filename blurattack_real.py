#!/usr/bin/env python
# coding: utf-8

# In[7]:
import os
# import copy
import shutil
import foolbox
import numpy as np
from fmodel import create_fmodel
from bmodel import create_bmodel
from utils import read_images, store_adversarial, load_adversarial, compute_MAD
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import warnings
import sys, getopt
from tqdm import tqdm
import math
import visdom
from PIL import Image
import cv2

warnings.filterwarnings("ignore")

test_model_acc = False


def run_attack_mifgsm(model, image, label, pert_type,imgname,eplison, blur_model, step_size=5,numSP=-1,mask_att_l1=2.0,direction=None):

    # apply the attack
    distance = foolbox.distances.Linfinity
    attack = foolbox.attacks.MomentumIterativeAttack(model,distance=distance)
    print(pert_type)
    binary_search = False
    adversarial = attack(image, label,
                         binary_search=binary_search,
                         epsilon=eplison,
                         stepsize=step_size,
                         iterations=10,
                         random_start=False,
                         return_early=True,
                         unpack=False,
                         pert_type= pert_type,
                         blur_model = blur_model,
                         numSP = numSP,
                         mask_att_l1=mask_att_l1,
                         direction=direction,
                         imgname = imgname)

    advs = [a.perturbed for a in adversarial]
    advs = [
        p if p is not None else np.full_like(u, np.nan)
        for p, u in zip(advs, image)
    ]
    perturbed_image = np.stack(advs)

    diff = np.linalg.norm(perturbed_image - image)
    if diff ==0:
        status =0
    else:
        if adversarial[0].adversarial_class is not None:
            status = 1
        else:
            status = -1

    return perturbed_image, status

# ------- testing on dev dataset -----------------------#
def test_real(model, white_model_name,attack_func, bbmodel_names, method, att_type, data_dir,eplison = np.array([0.5,51]) ,
             blur_strategy=None, step_size=5, numSP=-1,mask_att_l1=2.0,direction = None,deblurred = None,gpuid=0):

    method_name = white_model_name+"_"+white_model_name+"_"+method
    result_root ="put project path here"

    if method[0:5] == "mbAdv":
        pert_type = "Blur"
        step_size = int(step_size)
    else:
        pert_type = "Add"

    if not os.path.exists(result_root+"results/real/"):
        os.mkdir(result_root+"results/real/")

    if not os.path.exists(result_root+"results/real/"+method_name):
        os.mkdir(result_root+"results/real/"+method_name)

    valdir = os.path.join(data_dir)

    batch_size = 1

    if len(eplison)==2:
        if numSP==-1 or blur_strategy not in ["bg_obj_att","obj_att","att"]:
            file_name = 'real/' + method_name + '/eplison_{}_{}'.format(eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}/'.format(blur_strategy)
        elif numSP==-3:
            file_name = 'real/' + method_name + '/eplison_{}_{}'.format(eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}'.format(blur_strategy)+'_mask_att_l1_{}/'.format(mask_att_l1)

        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    elif len(eplison)==1:
        file_name = 'real/' + method_name + '/eplison_{}'.format(eplison[0])+'_stepsize_{}'.format(step_size)+'_blur_strategy_{}/'.format(blur_strategy)
        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    if direction is not None:
        file_name = 'real/' + method_name + '/eplison_{}_{}'.format(eplison[0], eplison[1]) + '_stepsize_{}'.format(
            step_size) + '_direction_{}_{}'.format(direction[0],direction[1])+'_blur_strategy_{}/'.format(blur_strategy)
        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    print("savename:{}".format(file_name))

    if isinstance(eplison,np.ndarray) and len(eplison)==1:
        eplison = eplison[0]

    slt_num = 7
    val_loader = torch.utils.data.DataLoader(
        datasets.Real(valdir, transform = transforms.Compose([transforms.CenterCrop([1080,1080]), transforms.Resize(299), transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False)
    piltransform = transforms.Compose([transforms.CenterCrop([1080,1080]), transforms.Resize(299), transforms.ToTensor()])

    success_status = np.ones([slt_num])*-1.
    success_status_fmodels = []
    fb_models = []
    checkexist = True

    if deblurred is not None:
        direct_eval = True
        file_name = file_name[:-1]+"_"+deblurred+"/"
        print(file_name)
    else:
        direct_eval = False

    if bbmodel_names is not None:
        for forward_model_name in bbmodel_names:
            success_status_fmodels.append(np.ones([slt_num])*-1.)
            forward_model = create_fmodel("imagenet", model_name=forward_model_name, gpu=gpuid)
            fb_models.append(forward_model)

    vis = visdom.Visdom()

    for i, (images, true_labels, target_labels, index, sample_path) in enumerate(tqdm(val_loader)):

        if i <3:
            continue

        file_path,file_full_name = os.path.split(sample_path[0])
        image_name, ext = os.path.splitext(file_full_name)

        file_name_ = file_name + image_name
        index = index.numpy()[0]

        print("Processing:" + file_name_)
        # try:
        images = images.numpy()

        # predict the original label
        predictions = model.forward_one(images.squeeze(0))
        label_or_target_class = np.array([np.argmax(predictions)])

        # apply the attack
        if torch.is_tensor(images):
            images = images.numpy()  # .squeeze(0).permute(1, 2, 0).numpy()
        vis.images(images,win='org')

        adversarial, success_status[index] = attack_func(model, images, label_or_target_class, pert_type,
                                                         os.path.join(valdir+"_saliency", image_name + "_saliency.jpg"),
                                                         eplison, blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction)

        # generate real-blur image

        vid_name, frameid = image_name.split('_')
        frameid = int(frameid)
        video_path = file_path+'_video/'+vid_name+'/'
        imgs = []
        for imageid in range(frameid-10,frameid+10):
            framepath = video_path+'{0:04d}.jpg'.format(imageid)
            img = Image.open(framepath).convert('RGB')
            if piltransform is not None:
                img = piltransform(img)
            img = img.numpy()
            imgs.append(img)
        realblur = np.array(imgs).mean(axis=0)
        # realblur = realblur.astype(np.float32).transpose(2, 0, 1)
        vis.image(realblur,win='realblur')

        # do blackbox attack
        if success_status[index] == 1:
            vis.images(adversarial,win='advblur')
            # predict the original label
            predictions = model.forward_one(adversarial.squeeze(0))
            advblur_class = np.array([np.argmax(predictions)])
            print("advblur_cls:{}".format(advblur_class))
            predictions = model.forward_one(realblur)
            realblur_class = np.array([np.argmax(predictions)])
            print("realblur_cls:{}".format(realblur_class))

            if adversarial.max() > 1:
                adversarial = adversarial/ 255
            adversarial = adversarial.astype("float32")
            store_adversarial(file_name_, images, adversarial)


    np.save(result_root+"results/" + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, white_model_name, slt_num), success_status)
    k=0
    for forward_model_name in bbmodel_names:
        np.save(result_root+"results/" + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, forward_model_name, slt_num),
            success_status_fmodels[k])
        k+=1

    print("\n", method_name, "\n")

def main(argv):

    opts, args = getopt.getopt(sys.argv[1:], "t:d:m:g:i:w:e:b:s:n:a:r:u:", ["attack_type","dataset", "method_name","gpu_id","ifdobb",
                                                                  "white_model_name","eplison","blur_strategy",
                                                                  "step_size","numSP","mask_att_l1","direction","deblurred"])
    gpu_id = 0
    method_name = "mbAdv_bim"
    ifdobb = 0
    white_model_name = "inceptionv3"
    dataset = "real"
    attack_type = "UA"
    eplison = np.array([0.1,30])
    direction = np.array([40,40])
    blur_strategy = "umot_whole"
    step_size = 20
    numSP = -2
    mask_att_l1 = 2.0
    deblurred = None

    for op, value in opts:
        if op == '-t' or op == '--attack_type':
            attack_type = value
        if op == '-d' or op == '--dataset':
            dataset = value
        if op == '-g' or op == '--gpu_id':
            gpu_id = value
        if op == '-m' or op == '--method':
            method_name = value
        if op == '-i' or op == '--ifdobb':
            ifdobb = value
        if op == '-w' or op == '--white_model_name':
            white_model_name = value
        if op == '-e' or op == '--eplison':
            eplison=np.array(value.split(',')).astype(float)
        if op == '-b' or op == '--blur_strategy':
            blur_strategy = value
        if op == '-s' or op == '--step_size':
            step_size = float(value)
        if op == '-n' or op == '--numsp':
            numSP = float(value)
        if op == '-a' or op == '--mask_att_l1':
            mask_att_l1 = float(value)
        if op == '-r' or op == '--direction':
            direction=np.array(value.split(',')).astype(float)
            print(direction)
        if op == '-u' or op == '--deblurred':
            if value == "None":
                deblurred = None
            else:
                deblurred = value

    if dataset == 'real':
        dataset_path = "./datasets/real"

    print('dataset path:{}'.format(dataset_path))
    print('gpu id:{}'.format(gpu_id))

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)


    params = None
    model = create_bmodel(dataset,model_name=white_model_name, gpu=gpu_id,params=params)
    if ifdobb:
        bbmodel_names = ["inceptionresnetv2","inceptionv3","inceptionv4","xception"]
        bbmodel_names.remove(white_model_name)
    else:
        bbmodel_names = None

    print("\n\nStart Test...")
    test_real(model,white_model_name,run_attack_mifgsm,bbmodel_names, method_name, attack_type, dataset_path,
              eplison,blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction,deblurred = deblurred,gpuid=gpu_id)
    sys.exit(0)

if __name__ == '__main__':
    main(sys.argv)
