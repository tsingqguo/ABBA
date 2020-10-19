#!/usr/bin/env python
# coding: utf-8

# In[7]:
import os
import foolbox
import numpy as np
from fmodel import create_fmodel
from bmodel import create_bmodel
from utils import load_adversarial, save_adversarial
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import warnings
import sys, getopt
from tqdm import tqdm
warnings.filterwarnings("ignore")
test_model_acc = False


# random select slt_num images from the whole dataset
def slt_images(model, valdir, slt_num, slt_images_prior=None):
    if slt_images_prior is not None:
        valid_sampler = torch.utils.data.SubsetRandomSampler(slt_images_prior.tolist())
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, sampler=valid_sampler)
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True)

    slt_num_each = slt_num / 1000
    classes_num = np.zeros([1000])
    slt_images_idx = np.zeros([slt_num], dtype=int)

    k = 0
    for i, (images, labels, index, pathname) in enumerate(tqdm(val_loader)):
        label_ = labels.numpy()[0]

        if classes_num[label_] < slt_num_each:

            if slt_images_prior is not None:
                classes_num[label_] += 1
                slt_images_idx[k] = index.numpy()[0]
                k += 1
            else:
                if torch.is_tensor(images):
                    images = images.squeeze(0).permute(1, 2, 0).numpy()
                predictions = model.predictions(images)
                criterion1 = foolbox.criteria.Misclassification()
                is_adversarials = criterion1.is_adversarial(predictions, labels)

                if not is_adversarials:
                    classes_num[label_] += 1
                    slt_images_idx[k] = index.numpy()[0]
                    k += 1

    return slt_images_idx

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
def test_dev(model, white_model_name,attack_func, bbmodel_names, method, att_type, data_dir,eplison = np.array([0.5,51]) ,
             blur_strategy=None, step_size=5, numSP=-1,mask_att_l1=2.0,direction = None, deblurred = None,gpuid=0):

    method_name = white_model_name+"_"+white_model_name+"_"+method

    result_root = ''

    if method[0:5] == "mbAdv":
        pert_type = "Blur"
        step_size = int(step_size)
    else:
        pert_type = "Add"

    if not os.path.exists(result_root):
        os.mkdir(result_root)

    result_root = result_root + '/' + method_name + '/'

    if not os.path.exists(result_root):
        os.mkdir(result_root)

    valdir = os.path.join(data_dir)

    batch_size = 1

    print("eplison:{}".format(eplison))

    if att_type == 'TA':
        file_att_type=att_type
    else:
        file_att_type = ''

    if len(eplison)==2:
        if numSP==-1 or blur_strategy not in ["bg_obj_att","obj_att","att"]:
            file_name = '/{}eplison_{}_{}'.format(file_att_type, eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}/'.format(blur_strategy)
        elif numSP==-3:
            file_name = '/{}eplison_{}_{}'.format(file_att_type, eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}'.format(blur_strategy)+'_mask_att_l1_{}/'.format(mask_att_l1)

        if not os.path.exists(result_root+ file_name):
            os.mkdir(result_root + file_name)

    elif len(eplison)==1:

        eplison[0] = np.round(eplison[0],4)
        step_size = np.round(step_size, 4)
        file_name = '/{}eplison_{}'.format(file_att_type, eplison[0])+'_stepsize_{}'.format(step_size)+'_blur_strategy_{}/'.format(blur_strategy)

        if not os.path.exists(result_root + file_name):
            os.mkdir(result_root + file_name)

    print(file_name)

    if direction is not None:
        file_name = '/eplison_{}_{}'.format(eplison[0], eplison[1]) + '_stepsize_{}'.format(
            step_size) + '_direction_{}_{}'.format(direction[0],direction[1])+'_blur_strategy_{}/'.format(blur_strategy)
        if not os.path.exists(result_root + file_name):
            os.mkdir(result_root + file_name)

    print("savename:{}".format(file_name))

    if isinstance(eplison,np.ndarray) and len(eplison)==1:
        eplison = eplison[0]


    slt_num = 1000
    val_loader = torch.utils.data.DataLoader(
        datasets.Dev(valdir, target_file='dev_dataset.csv', transform = transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False)

    success_status = np.ones([slt_num])*-1.
    success_status_fmodels = []
    fb_models = []
    checkexist = True

    for forward_model_name in bbmodel_names:
        success_status_fmodels.append(np.ones([slt_num])*-1.)
        forward_model = create_fmodel("imagenet", model_name=forward_model_name, gpu=gpuid)
        fb_models.append(forward_model)

    for i, (images, true_labels, target_labels, index, sample_path) in enumerate(tqdm(val_loader)):

        file_path,file_full_name = os.path.split(sample_path[0])
        image_name, ext = os.path.splitext(file_full_name)
        file_name_ = file_name + image_name
        index = index.numpy()[0]

        if os.path.exists(os.path.join(result_root, file_name_+".png")) and checkexist:
            success_status[index], original, adversarial = load_adversarial(file_name_, images)
            print(file_name_+" exists!")

        if att_type=="TA":
            label_or_target_class = target_labels.numpy()
        else:
            label_or_target_class = true_labels.numpy()

        if torch.is_tensor(images):
            images = images.numpy()

        adversarial, success_status[index] = attack_func(model, images, label_or_target_class, pert_type,
                                                         os.path.join(valdir+"_saliency", image_name + "_saliency.jpg"),
                                                         eplison, blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction)

        save_adversarial(result_root+file_name_, adversarial)


    np.save(result_root + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, white_model_name, slt_num), success_status)
    k=0
    for forward_model_name in bbmodel_names:
        np.save(result_root + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, forward_model_name, slt_num),
            success_status_fmodels[k])
        k+=1

    print("\n", method_name, "\n")



# ------- testing on imagenet dataset -----------------------#
def test_imagenet(model, white_model_name,attack_func, bbmodel_names, method, att_type, data_dir,eplison = np.array([0.5,51]) ,
             blur_strategy=None, step_size=5, numSP=-1,mask_att_l1=2.0,direction = None, deblurred = None,gpuid=0):

    method_name = white_model_name+"_"+white_model_name+"_"+method

    result_root = ''

    if method[0:5] == "mbAdv":
        pert_type = "Blur"
        step_size = int(step_size)
    else:
        pert_type = "Add"

    if not os.path.exists(result_root):
        os.mkdir(result_root)

    result_root = result_root + '/' + method_name + '/'

    if not os.path.exists(result_root):
        os.mkdir(result_root)

    valdir = os.path.join(data_dir)

    batch_size = 1

    print("eplison:{}".format(eplison))

    if att_type == 'TA':
        file_att_type=att_type
    else:
        file_att_type = ''

    if len(eplison)==2:
        if numSP==-1 or blur_strategy not in ["bg_obj_att","obj_att","att"]:
            file_name = '/{}eplison_{}_{}'.format(file_att_type, eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}/'.format(blur_strategy)
        elif numSP==-3:
            file_name = '/{}eplison_{}_{}'.format(file_att_type, eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}'.format(blur_strategy)+'_mask_att_l1_{}/'.format(mask_att_l1)

        if not os.path.exists(result_root+ file_name):
            os.mkdir(result_root + file_name)

    elif len(eplison)==1:

        eplison[0] = np.round(eplison[0],4)
        step_size = np.round(step_size, 4)
        file_name = '/{}eplison_{}'.format(file_att_type, eplison[0])+'_stepsize_{}'.format(step_size)+'_blur_strategy_{}/'.format(blur_strategy)

        if not os.path.exists(result_root + file_name):
            os.mkdir(result_root + file_name)

    print(file_name)

    if direction is not None:
        file_name = '/eplison_{}_{}'.format(eplison[0], eplison[1]) + '_stepsize_{}'.format(
            step_size) + '_direction_{}_{}'.format(direction[0],direction[1])+'_blur_strategy_{}/'.format(blur_strategy)
        if not os.path.exists(result_root + file_name):
            os.mkdir(result_root + file_name)

    print("savename:{}".format(file_name))

    if isinstance(eplison,np.ndarray) and len(eplison)==1:
        eplison = eplison[0]

    # define the dataloader
    #------------------------------------------------------------------------------------------------------------------#
    workers = 4
    slt_num = 1000
    slt_name = result_root+"results/imagenet_slt_" + str(slt_num) + ".npy"

    slt_num_saved = 10000
    slt_name_saved = result_root+"results/imagenet_slt_" + str(slt_num_saved) + ".npy"

    if os.path.exists(slt_name):
        sltIdx = np.load(slt_name)
        sltIdx.sort(axis=0)
    else:
        if os.path.exists(slt_name_saved) and slt_num_saved>=slt_num:
            sltIdx_saved = np.load(slt_name_saved)
            sltIdx = slt_images(model, valdir, slt_num,sltIdx_saved)
            sltIdx.sort(axis=0)
            np.save(slt_name, sltIdx)
        else:
            # slt images from imagenet
            sltIdx = slt_images(model,valdir,slt_num)
            #sltIdx = np.random.choice(50000,slt_num,replace=False)
            sltIdx.sort(axis=0)
            np.save(slt_name, sltIdx)

    valid_sampler = torch.utils.data.SubsetRandomSampler(sltIdx.tolist())
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=valid_sampler)
    #------------------------------------------------------------------------------------------------------------------#
    success_status = np.ones([slt_num])*-1.
    success_status_fmodels = []
    fb_models = []
    checkexist = True

    for forward_model_name in bbmodel_names:
        success_status_fmodels.append(np.ones([slt_num])*-1.)
        forward_model = create_fmodel("imagenet", model_name=forward_model_name, gpu=gpuid)
        fb_models.append(forward_model)

    for i, (images, true_labels, target_labels, index, sample_path) in enumerate(tqdm(val_loader)):

        file_path,file_full_name = os.path.split(sample_path[0])
        image_name, ext = os.path.splitext(file_full_name)
        file_name_ = file_name + image_name
        index = index.numpy()[0]

        if os.path.exists(os.path.join(result_root, file_name_+".png")) and checkexist:
            success_status[index], original, adversarial = load_adversarial(file_name_, images)
            print(file_name_+" exists!")

        if att_type=="TA":
            label_or_target_class = target_labels.numpy()
        else:
            label_or_target_class = true_labels.numpy()

        if torch.is_tensor(images):
            images = images.numpy()

        adversarial, success_status[index] = attack_func(model, images, label_or_target_class, pert_type,
                                                         os.path.join(valdir+"_saliency", image_name + "_saliency.jpg"),
                                                         eplison, blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction)

        save_adversarial(result_root+file_name_, adversarial)


    np.save(result_root + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, white_model_name, slt_num), success_status)
    k=0
    for forward_model_name in bbmodel_names:
        np.save(result_root + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, forward_model_name, slt_num),
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
    dataset = "imagenet"
    attack_type = "UA"
    eplison = np.array([0.65,15])
    direction = None
    blur_strategy = "joint"
    step_size = 20
    numSP = -1
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

    if dataset == 'imagenet':
        dataset_path = "/mnt/nvme/dataset/ILSVRC2012/"
    elif dataset == 'deblur':
        dataset_path = "/mnt/nvme/dataset/deblur/"
    elif dataset == 'dev':
        dataset_path = "/mnt/nvme/dataset/dev/images"

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

    result_root ="/mnt/nvme/projects/BlurAttack/"

    if dataset == "dev":
        test_dev(result_root, model, white_model_name, run_attack_mifgsm,bbmodel_names, method_name, attack_type, dataset_path,
                     eplison,blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction,deblurred = deblurred,gpuid=gpu_id)
    elif dataset == "imagenet":
        test_imagenet(result_root, model, white_model_name, run_attack_mifgsm,bbmodel_names, method_name, attack_type, dataset_path,
                     eplison,blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction,deblurred = deblurred,gpuid=gpu_id)

    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
