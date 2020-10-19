#!/usr/bin/env python
# coding: utf-8

# This code is for the experimental results in supplementary material for interpretation of the high transferability.
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
from torch.nn import functional as F
import cv2 as cv
import visdom
from torch.autograd import Variable

warnings.filterwarnings("ignore")


def save(mask, img, blurred,dir):
    mask = mask.cpu().data.numpy()[0]
    #mask = np.transpose(mask, (1, 2, 0))
    img = img.transpose(2,3,1,0).squeeze(-1)
    blurred = blurred.transpose(2,3,1,0).squeeze(-1)

    #mask = (mask - np.min(mask)) / np.max(mask)
    heatmap = cv.applyColorMap(np.uint8(255 * mask), cv.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = 1. * heatmap + np.float32(img)
    cam = cam / np.max(cam)

    img = np.float32(img)
    perturbated = np.multiply(mask, img) + np.multiply(1-mask, blurred)

    cv.imwrite(dir+"perturbated.png", np.uint8(255 * perturbated))
    cv.imwrite(dir+"heatmap.png", np.uint8(255 * heatmap))
    cv.imwrite(dir+"mask.png", np.uint8(255 * mask))
    cv.imwrite(dir+"cam.png", np.uint8(255 * cam))

def numpy_to_torch(img, requires_grad = True):
	if len(img.shape) < 3:
		output = np.float32([img])
	else:
		output = np.transpose(img, (2, 0, 1))

	output = torch.from_numpy(output).cuda()

	output.unsqueeze_(0)
	v = Variable(output, requires_grad = requires_grad)
	return v

def interp_explan(original, pertured, category, model,save_path):

    max_iterations = 50#100
    tv_beta = 3
    learning_rate = 0.1
    l1_coeff = 0.05
    tv_coeff = 0.2
    mask_org = torch.zeros([28, 28]).cuda()
    mask_org.requires_grad_()
    optimizer = torch.optim.Adam([mask_org], lr=learning_rate)

    original = torch.from_numpy(original).cuda()
    pertured = torch.from_numpy(pertured).cuda()

    outputs_inti_pertured = torch.nn.Softmax()(model._model(pertured))

    _, pertured_labels = torch.topk(outputs_inti_pertured[0,:],2)
    pertured_labels = pertured_labels.cpu().detach().numpy()
    print("gt_label:{}, pertured_label:{}".format(category,pertured_labels))

    if pertured_labels[0] != category:
        diff_conf = outputs_inti_pertured[0,pertured_labels[0]] - outputs_inti_pertured[0,category]
        diff_conf = diff_conf.cpu().detach().numpy()
    else:
        diff_conf = outputs_inti_pertured[0,pertured_labels[1]] - outputs_inti_pertured[0,category]
        diff_conf = diff_conf.cpu().detach().numpy()

    print(save_path+"mask.png")
    if os.path.exists(save_path+"mask.png"):
        mask = cv.imread(save_path+"mask.png")
        mask = torch.from_numpy(mask/255).cuda()
        mask = mask.unsqueeze(0).permute(0,3,1,2)
        # Use the mask to perturbated the input image.
        perturbated_input = original.mul(1 - mask) + \
                            pertured.mul(mask)
        mask = mask.permute(0,2,3,1)
        print("mask file exists!")
        return perturbated_input, mask, diff_conf

    def tv_norm(input, tv_beta):
        img = input
        row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
        col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
        return row_grad + col_grad

    print("start iteration")
    for i in range(max_iterations):
        #print("interp_iter:{}".format(i))
        mask = F.upsample(mask_org.unsqueeze(0).unsqueeze(0), (299, 299), mode='bilinear')
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        mask = mask.squeeze(0).repeat(3, 1, 1)
        # Use the mask to perturbated the input image.
        perturbated_input = original.mul(1 - mask) + \
                            pertured.mul(mask)

        outputs = torch.nn.Softmax()(model._model(perturbated_input))

        loss = l1_coeff * torch.mean(torch.abs(mask)) + \
               tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

        # cam = 1.0 * mask.unsqueeze(0) + original
        # cam = (cam-cam.min())/(cam.max()-cam.min())
        # vis = visdom.Visdom(env='Adversarial Example Showing')
        # vis.images(original, win='original')
        # vis.images(pertured, win='blurred')
        # vis.images(mask, win='mask')
        # vis.images(cam, win='cam')

    mask_ = F.upsample(mask.unsqueeze(0), (299, 299), mode='bilinear').permute(0, 2, 3, 1)

    return perturbated_input, mask_, diff_conf

# ------- testing on dev dataset -----------------------#
def test_dev(model, white_model_name,bbmodel_names, method, data_dir,eplison = np.array([0.5,51]) ,
             blur_strategy=None, step_size=5, numSP=-1,mask_att_l1=2.0,gpuid=0):

    method_name = white_model_name+"_"+white_model_name+"_"+method
    result_root =  "put project path here"

    if not os.path.exists(result_root+"results/dev/"):
        os.mkdir(result_root+"results/dev/")

    if not os.path.exists(result_root+"results/dev/"+method_name):
        os.mkdir(result_root+"results/dev/"+method_name)

    valdir = os.path.join(data_dir)

    batch_size = 1

    if len(eplison)==2:
        if numSP==-1 or blur_strategy not in ["bg_obj_att","obj_att","att"]:
            file_name = 'dev/' + method_name + '/eplison_{}_{}'.format(eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}/'.format(blur_strategy)
        elif numSP==-3:
            file_name = 'dev/' + method_name + '/eplison_{}_{}'.format(eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}'.format(blur_strategy)+'_mask_att_l1_{}/'.format(mask_att_l1)

        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    elif len(eplison)==1:
        file_name = 'dev/' + method_name + '/eplison_{}'.format(eplison[0])+'_stepsize_{}'.format(step_size)+'_blur_strategy_{}/'.format(blur_strategy)
        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    print("savename:{}".format(file_name))

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

    save_dir = result_root+ "/experiments/interp_results/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = result_root+"/experiments/interp_results/"+ method_name+"/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = save_dir + '/eplison_{}'.format(eplison[0])+'_stepsize_{}'.format(step_size)+'_blur_strategy_{}/'.format(blur_strategy)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # success rate
    succ_rate = np.ones([slt_num])*-10.0
    consist_sum = np.ones([slt_num])*-10.0
    consist_max = np.ones([slt_num])*-10.0


    for i, (images, true_labels, target_labels, index, sample_path) in enumerate(tqdm(val_loader)):

        file_path,file_full_name = os.path.split(sample_path[0])
        image_name, ext = os.path.splitext(file_full_name)
        file_name_ = file_name + image_name
        index = index.numpy()[0]

        if os.path.exists(os.path.join(result_root+"results", file_name_+".npy")) and checkexist:
            success_status[index], original, adversarial = load_adversarial(file_name_,images)
            print(file_name_+" exists!")

            # do blackbox attack
            adv_masks = []
            diff_confs = []
            if success_status[index] == 1:
                # generate inpertation for the mbAdv_mifgsm
                save_path = save_dir + image_name + white_model_name
                adv_exp, adv_mask, diff_conf = interp_explan(original, adversarial, true_labels.numpy()[0], model,save_path)
                save(adv_mask, original, adversarial, save_dir + image_name + white_model_name)
                adv_mask = adv_mask.cpu().detach().numpy()
                adv_mask = adv_mask[:,:,:,0].transpose(1,2,0)
                adv_masks.append(adv_mask)
                diff_confs.append(diff_conf)
                if adversarial.max() > 1:
                    adversarial = adversarial / 255
                adversarial = adversarial.astype("float32")
                k = 0
                for forward_model in fb_models:
                    save_path = save_dir + image_name + bbmodel_names[k]
                    adv_exp,adv_mask, diff_conf = interp_explan(original,adversarial,true_labels.numpy()[0],forward_model,save_path)
                    save(adv_mask, original, adversarial, save_dir + image_name+bbmodel_names[k])
                    k+=1
                    adv_mask = adv_mask.cpu().detach().numpy()
                    adv_mask = adv_mask[:, :, :, 0].transpose(1, 2, 0)
                    adv_masks.append(adv_mask)
                    diff_confs.append(diff_conf)
                # calculate the consistency
                adv_masks=np.concatenate(adv_masks, axis=2)
                consist_sum[i] = adv_masks.std(axis=2).sum()
                consist_max[i] = adv_masks.std(axis=2).max()
                # calculate the success rate
                succ_rate[i] = np.array(diff_confs).mean()
                #
                print("consist_sum:{}consist_max:{} succ_rate:{}".format(consist_sum[i],consist_max[i],succ_rate[i]))
        else:
            print("continue!")
            continue
    np.savez(save_dir+"/consist_succ.npz",consist_sum,consist_max,succ_rate)
    print("Processing:" + file_name_)

def main(argv):

    opts, args = getopt.getopt(sys.argv[1:], "d:m:g:i:w:e:b:s:n:a:", ["dataset", "method_name","gpu_id","ifdobb",
                                                                  "white_model_name","eplison","blur_strategy",
                                                                  "step_size","numSP","mask_att_l1"])
    gpu_id = 0
    method_name = "mbAdv_bim"
    ifdobb = 1
    white_model_name = "inceptionv3"
    dataset = "dev"
    eplison = np.array([0.5,10])
    blur_strategy = "joint"
    step_size = 10
    numSP = -1
    mask_att_l1 = 2.0

    for op, value in opts:
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

    if dataset == 'dev':
        dataset_path = "/home/wangjian/tsingqguo/dataset/dev/images"

    print('dataset path:{}'.format(dataset_path))
    print('gpu id:{}'.format(gpu_id))

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)

    model = create_bmodel(dataset,model_name=white_model_name, gpu=gpu_id)

    if ifdobb:
        bbmodel_names = ["inceptionresnetv2","inceptionv3","inceptionv4","xception"]
        bbmodel_names.remove(white_model_name)
    else:
        bbmodel_names = None

    test_dev(model, white_model_name,bbmodel_names, method_name, dataset_path,
                     eplison, blur_strategy, step_size,numSP=numSP,mask_att_l1=mask_att_l1,gpuid=gpu_id)
    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
