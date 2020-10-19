import numpy as np
from abc import abstractmethod
import logging
import warnings
import os
from .base import Attack
from .base import generator_decorator
from .. import distances
from ..utils import crossentropy
from .. import nprng
from ..optimizers import AdamOptimizer
from ..optimizers import GDOptimizer
import torch
from torch.nn import functional as F
import visdom
import cv2 as cv
# for Poolnet saliency detection
from tools.saliency.networks.poolnet import build_model, weights_init
import math


class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasbses should implement as_generator, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _create_optimizer(self, a, stepsize):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(
        self,
        a,
        binary_search,
        epsilon,
        stepsize,
        iterations,
        random_start,
        return_early,
        gradient_args={},
        pert_type = "Add",
        blur_model="joint",
        numSP = -1,
        mask_att_l1 = 2.0,
        direction = None,
        imgname = None
    ):
        self.imgname = imgname
        self.blur_model = blur_model
        self.kernel_size = 51
        self.numSP = numSP
        self.mask_att_l1 = mask_att_l1
        self.direction = direction
        if not a.has_gradient():
            warnings.warn(
                "applied gradient-based attack to model that"
                " does not provide gradients"
            )
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            yield from self._run_binary_search(
                a,
                epsilon,
                stepsize,
                iterations,
                random_start,
                targeted,
                class_,
                return_early,
                k=k,
                gradient_args=gradient_args,
                pert_type=pert_type,
            )
            return
        else:
            optimizer = self._create_optimizer(a, stepsize)

            success = yield from self._run_one(
                a,
                epsilon,
                optimizer,
                iterations,
                random_start,
                targeted,
                class_,
                return_early,
                gradient_args,
                pert_type
            )
            return success

    def _run_binary_search(
        self,
        a,
        epsilon,
        stepsize,
        iterations,
        random_start,
        targeted,
        class_,
        return_early,
        k,
        gradient_args,
        pert_type="Add",
    ):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            optimizer = self._create_optimizer(a, stepsize)

            success = yield from self._run_one(
                a,
                epsilon,
                optimizer,
                iterations,
                random_start,
                targeted,
                class_,
                return_early,
                gradient_args,
                pert_type,
            )
            return success

        for i in range(k):
            success = yield from try_epsilon(epsilon)
            if success:
                logging.info("successful for eps = {}".format(epsilon))
                break
            logging.info("not successful for eps = {}".format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning("exponential search failed")
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            success = yield from try_epsilon(epsilon)
            if success:
                good = epsilon
                logging.info("successful for eps = {}".format(epsilon))
            else:
                bad = epsilon
                logging.info("not successful for eps = {}".format(epsilon))

    def _run_one(
        self,
        a,
        epsilon,
        optimizer,
        iterations,
        random_start,
        targeted,
        class_,
        return_early,
        gradient_args,
        pert_type="Add"
    ):
        """ Modified the _run_one() to add the motion-blur-aware attack.
            pert_type = "Add" means to use the trainditional additional noise
                      = "Blur" means to use the novel motion-blur-aware attack
            kernel_size : define the size of linear kernel for motion blur
            blur_model: we define four attacking models:
                        image-level motion blur: whole
                        object-aware motion blur: obj
                        background-aware motion blur: backg
                        joint-object-background motion blur: joint
            """
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.unperturbed.copy()

        self.disp = False

        if pert_type is "Add":
            if random_start:
                # using uniform noise even if the perturbation clipping uses
                # a different norm because cleverhans does it the same way
                noise = nprng.uniform(-epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype
                )
                x = original + self._clip_perturbation(a, noise, epsilon)

                strict = False  # because we don't enforce the bounds here
            else:
                x = original
                strict = True

            success = False
            for _ in range(iterations):
                gradient = yield from self._gradient(
                    a, x, class_, strict=strict, gradient_args=gradient_args
                )
                # non-strict only for the first call and
                # only if random_start is True

                strict = True
                if not targeted:
                    gradient = -gradient

                # untargeted: gradient ascent on cross-entropy to original class
                # targeted: gradient descent on cross-entropy to target class
                if self.numSP == -3:

                    kernel_sz = self.kernel_size
                    theta_f, theta_b, alpha, mask = self.init_flow_alpha(original, kernel_sz, "default")

                    if not hasattr(self,"mask_att") and self.blur_model == "att":
                        if os.path.exists(self.imgname + "_att{}.npy".format(self.mask_att_l1)):
                            mask_att = np.load(self.imgname + "_att{}.npy".format(self.mask_att_l1))
                            mask_att = torch.from_numpy(mask_att).cuda()
                            print("mask_att. loaded!")
                        else:
                            mask_att = self.adapt_mask(original, mask, class_, a._model)
                            # make sure the 1 number of mask_att is smaller than that of mask
                            mask_diff = mask - mask_att
                            mask_att[mask_diff < 0] = 0
                            np.save(self.imgname + "_att{}.npy".format(self.mask_att_l1), mask_att.detach().cpu().numpy())
                            print("mask_att. calculated!")
                        self.mask_att = mask_att

                    if self.blur_model == "obj":

                        pred = mask.squeeze(-1).cpu().numpy()
                        gradient_f = (pred) * gradient
                        x_f = pred*x
                        x_b = (1 - pred) * x
                        x_f = x_f + optimizer(gradient_f)
                        x = x_f + x_b

                    elif self.blur_model=="att":

                        pred = mask.squeeze(-1).cpu().numpy()
                        pred_att = mask_att.squeeze(-1).cpu().numpy()
                        gradient_f_att = (pred_att) * gradient
                        x_f_att = pred_att * x
                        x_f = (pred-pred_att)*x
                        x_b = (1 - pred) * x
                        x_f_att = x_f_att + optimizer(gradient_f_att)
                        x = x_f + x_b + x_f_att

                    elif self.blur_model=="bg":

                        pred = mask.squeeze(-1).cpu().numpy()
                        gradient_b = (1 - pred) * gradient
                        x_f = (pred)*x
                        x_b = (1 - pred) * x
                        x_b = x_b + optimizer(gradient_b)
                        x = x_f + x_b

                else:

                    x = x + optimizer(gradient)


                x = original + self._clip_perturbation(a, x - original, epsilon)

                x = np.clip(x, min_, max_)

                logits, is_adversarial = yield from a.forward_one(x)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        logging.debug(
                            "crossentropy to {} is {}".format(a.original_class, ce)
                        )
                    ce = crossentropy(class_, logits)
                    logging.debug("crossentropy to {} is {}".format(class_, ce))

                if is_adversarial:
                    if return_early:
                        return True
                    else:
                        success = True

        elif pert_type == "Blur":

            kernel_sz = self.kernel_size

            if self.numSP == -2: # unified motion
                self.alpha_len = epsilon[1]
                self.trans_val = epsilon[0]
            else:
                self.alpha_len = 1

            if random_start:
                # using uniform noise even if the perturbation clipping uses
                # a different norm because cleverhans does it the same way
                theta_f, theta_b, alpha = self.init_flow_alpha(original, kernel_sz, "rand_alpha")
                x = self.add_flow_alpha_pert(original, kernel_sz, theta_f, theta_b, alpha)
                x = np.clip(x, min_, max_)
                strict = False  # because we don't enforce the bounds here

            else:
                theta_f, theta_b, alpha, mask = self.init_flow_alpha(original, kernel_sz,"default")
                x = self.add_flow_alpha_pert(original, kernel_sz, theta_f, theta_b, alpha,mask)
                x = np.clip(x, min_, max_)
                strict = True

            if self.disp:
                import visdom
                vis = visdom.Visdom(env='Adversarial Example Showing')
                vis.images(torch.from_numpy(original), win='X_org')

            success = False

            # calulate the mask directly
            if self.numSP==-3:
                self.blur_model == "bg_obj_att"
                if os.path.exists(self.imgname + "_att{}.npy".format(self.mask_att_l1)):
                    mask_att = np.load(self.imgname + "_att{}.npy".format(self.mask_att_l1))
                    mask_att = torch.from_numpy(mask_att).cuda()
                    print("mask_att. loaded!")
                else:
                    mask_att = self.adapt_mask_gaussblur(original, mask, class_, a._model)
                    # make sure the 1 number of mask_att is smaller than that of mask
                    mask_diff = mask - mask_att
                    mask_att[mask_diff < 0] = 0
                    np.save(self.imgname + "_att{}.npy".format(self.mask_att_l1),mask_att.detach().cpu().numpy())
                    print("mask_att. calculated!")

            for i in range(iterations):
                gradient = yield from self._gradient(
                    a, x, class_, strict=strict, gradient_args=gradient_args
                )

                theta_f = theta_f.detach()
                theta_b = theta_b.detach()
                alpha = alpha.detach()
                mask = mask.detach()

                grad_theta_f, grad_theta_b, grad_alpha, grad_mask = self.grad_flow_alpha_pert(original, kernel_sz, theta_f,
                                                                                   theta_b, alpha, mask, gradient)

                # non-strict only for the first call and
                # only if random_start is True
                strict = True

                if not targeted:
                    grad_theta_f = -grad_theta_f
                    grad_theta_b = -grad_theta_b
                    grad_alpha = -grad_alpha

                if self.numSP != -2 and self.blur_model != "joint_wo_adaptmot":
                    # updating theta_f and theta_b
                    theta_f[:, 2] = theta_f[:, 2] + 1e-2 * optimizer(grad_theta_f[:, 2])
                    theta_b[:, 2] = theta_b[:, 2] + 1e-2* optimizer(grad_theta_b[:, 2]) # nipsversion: 1e-2* optimizer(grad_theta_b[:, 2])

                # updating alpha:
                # using unified distribution of alpha
                if  self.numSP ==-2:
                    pred = mask.permute(0, 3, 1, 2)

                    alpha_f = pred * alpha
                    alpha_b = (1 - pred) * alpha

                    #alpha = alpha + optimizer(grad_alpha.mean())
                    if self.blur_model == "umot_whole":
                        alpha = alpha
                    elif self.blur_model == "umot_obj":
                        alpha_b = torch.zeros_like(alpha).cuda()
                        alpha_b[0, :, :, :] = 1.
                        alpha = alpha_b+alpha_f
                    elif self.blur_model == "umot_bg":
                        alpha_f = torch.zeros_like(alpha).cuda()
                        alpha_f[0, :, :, :] = 1.
                        alpha = alpha_b+alpha_f

                # object-saliecy level
                elif self.numSP == -3:

                    if self.blur_model == "bg_obj_att":

                        pred = mask.permute(0, 3, 1, 2)
                        pred_att = mask_att.permute(0, 3, 1, 2)
                        grad_alpha_f_att = (pred_att) * grad_alpha
                        grad_alpha_f = (pred-pred_att) * grad_alpha
                        grad_alpha_b = (1 - pred) * grad_alpha

                        alpha_f_att = pred_att * alpha
                        alpha_f = (pred-pred_att)*alpha
                        alpha_b = (1 - pred) * alpha

                        alpha_f = alpha_f + optimizer(grad_alpha_f.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))
                        alpha_f_att = alpha_f_att + optimizer(grad_alpha_f_att)
                        alpha_b = alpha_b + optimizer(grad_alpha_b.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))
                        alpha = alpha_f + alpha_b + alpha_f_att

                    elif self.blur_model=="obj_att":

                        pred = mask.permute(0, 3, 1, 2)
                        pred_att = mask_att.permute(0, 3, 1, 2)
                        grad_alpha_f_att = (pred_att) * grad_alpha
                        grad_alpha_f = (pred-pred_att) * grad_alpha

                        alpha_f_att = pred_att * alpha
                        alpha_f = (pred-pred_att)*alpha
                        alpha_b = (1 - pred) * alpha

                        alpha_f = alpha_f + optimizer(grad_alpha_f.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))
                        alpha_f_att = alpha_f_att + optimizer(grad_alpha_f_att)
                        alpha = alpha_f + alpha_b + alpha_f_att

                    elif self.blur_model=="bg_att":

                        pred = mask.permute(0, 3, 1, 2)
                        pred_att = mask_att.permute(0, 3, 1, 2)
                        grad_alpha_f_att = (pred_att) * grad_alpha
                        grad_alpha_b = (1 - pred) * grad_alpha

                        alpha_f_att = pred_att * alpha
                        alpha_f = (pred-pred_att)*alpha
                        alpha_b = (1 - pred) * alpha

                        alpha_f_att = alpha_f_att + optimizer(grad_alpha_f_att)
                        alpha_b = alpha_b + optimizer(grad_alpha_b.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))
                        alpha = alpha_f + alpha_b + alpha_f_att

                    elif self.blur_model=="att":
                        pred_att = mask_att.permute(0, 3, 1, 2)
                        grad_alpha_f_att = (pred_att) * grad_alpha

                        alpha_f_att = pred_att * alpha

                        alpha_f_att = alpha_f_att + optimizer(grad_alpha_f_att)
                        alpha = alpha_f + alpha_b + alpha_f_att


                # object level
                elif self.numSP==-1:

                    pred = mask.permute(0, 3, 1, 2)
                    grad_alpha_f = pred * grad_alpha
                    grad_alpha_b = (1 - pred) * grad_alpha

                    # Blur attack model attack selection
                    if self.blur_model == "whole":
                        grad_alpha_mean = (grad_alpha_f + grad_alpha_b) / 2
                        alpha = alpha + optimizer(grad_alpha_mean.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))

                    elif self.blur_model == "obj":

                        grad_alpha_b = torch.zeros_like(grad_alpha).cuda()

                        alpha_f = pred * alpha
                        alpha_b = (1 - pred) * alpha

                        alpha_f = alpha_f + optimizer(grad_alpha_f.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))
                        alpha_b = alpha_b + optimizer(grad_alpha_b.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))

                        alpha = alpha_f + alpha_b

                    elif self.blur_model == "bg":
                        grad_alpha_f = torch.zeros_like(grad_alpha).cuda()

                        alpha_f = pred * alpha
                        alpha_b = (1 - pred) * alpha

                        alpha_f = alpha_f + optimizer(grad_alpha_f.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))
                        alpha_b = alpha_b + optimizer(grad_alpha_b.view(kernel_sz, 1, 1, -1).mean(dim=3).unsqueeze(-1))

                        alpha = alpha_f + alpha_b

                # superpixel level
                elif self.numSP>0 and self.numSP<600 and hasattr(self,'superpixel'):

                    sp_tensor = self.sp_tensor
                    sp_pixelnum = self.sp_pixelnum

                    grad_alpha_sped = grad_alpha.unsqueeze(0)*sp_tensor/sp_pixelnum
                    grad_alpha = grad_alpha_sped.sum(dim=3).sum(dim=3).unsqueeze(-1).unsqueeze(-1)*sp_tensor
                    alpha = alpha + optimizer(grad_alpha.sum(0))

                # pixel level
                elif self.numSP>=600 :
                    alpha = alpha + optimizer(grad_alpha)

                if self.numSP !=-2 and not targeted:
                    # to constraint the 0-norm of alpha
                    #alpha[torch.topk(alpha, int(kernel_sz-epsilon[1]), dim=0, largest=False, sorted=False, out=None)[1]] = 0
                    min_alpha = alpha.min(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1)
                    max_alpha = alpha.max(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1)
                    alpha = (alpha - min_alpha) / (max_alpha - min_alpha + 1e-25)
                    alpha[int(epsilon[1]):, :, :, :] = 0
                    # to enforce the largest result at the first channel
                    alpha[0, :, :, :] = alpha.max(dim=0).values
                    alpha = alpha / (1e-25+alpha.sum(dim=0))

                # to constraint the shift values epsilon[0] range from -1 to 1
                theta_f[:, 2] = torch.clamp(theta_f[:, 2], -epsilon[0], epsilon[0])
                theta_b[:, 2] = torch.clamp(theta_b[:, 2], -epsilon[0], epsilon[0])

                # contraint the motion direction
                if self.direction is not None:
                    direction = self.direction*math.pi / 180
                    if direction[0] == 0:
                        print('fg:{} degrees!'.format(0))
                        theta_f[1, 2] = 0
                    elif self.direction[0] == 90:
                        print('fg:{} degrees!'.format(90))
                        theta_f[0,2] = 0
                    elif (self.direction[0]>45 and self.direction[0]<90) or (self.direction[0]<-45 and self.direction[0]>-90):
                        print('fg:{} degrees!'.format("45-90,-90--45"))
                        theta_f[0,2] = torch.clamp(theta_f[1,2]/torch.tan(torch.ones(1)*direction[0]),-epsilon[0], epsilon[0])
                    elif self.direction[0]<45 and self.direction[0]>-45:
                        print('fg:{} degrees!'.format("-45-45"))
                        theta_f[1,2] = theta_f[0,2]* torch.tan(torch.ones(1)*direction[0])

                    if direction[1] == 0:
                        print('bg:{} degrees!'.format(0))
                        theta_b[1, 2] = 0
                    elif self.direction[1] == 90:
                        print('bg:{} degrees!'.format(90))
                        theta_b[0,2] = 0
                    elif (self.direction[1]>45 and self.direction[1]<90) or (self.direction[1]<-45 and self.direction[1]>-90):
                        print('bg:{} degrees!'.format("45-90,-90--45"))
                        theta_b[0,2] = torch.clamp(theta_b[1,2]/torch.tan(torch.ones(1)*direction[1]),-epsilon[0], epsilon[0])
                    elif self.direction[1]<45 and self.direction[1]>-45:
                        print('bg:{} degrees!'.format("-45-45"))
                        theta_b[1,2] = theta_b[0,2]* torch.tan(torch.ones(1)*direction[1])

                x = self.add_flow_alpha_pert(original, kernel_sz, theta_f, theta_b, alpha, mask)
                x = np.clip(x, min_, max_)

                logits, is_adversarial = yield from a.forward_one(x)

                if self.disp :
                    vis.images(torch.from_numpy(x), win='adv_cls')
                    vis.images(torch.from_numpy(x - original), win='diff_cls')

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        logging.debug(
                            "crossentropy to {} is {}".format(a.original_class, ce)
                        )
                    ce = crossentropy(class_, logits)
                    logging.debug("crossentropy to {} is {}".format(class_, ce))

                if is_adversarial:

                    print("Adv theta params: theta_f: {} thetha_b: {}!".format(theta_f,theta_b))

                    if self.blur_model == "att" and pert_type == "Blur":

                        max_iterations = 500
                        model = a._model
                        category = a.original_class
                        tv_beta = 3
                        learning_rate = 0.05
                        l1_coeff = 0.05  # 0.05
                        tv_coeff = 0.1
                        mask = torch.zeros([28, 28]).cuda()
                        mask.requires_grad_()
                        optimizer = torch.optim.Adam([mask], lr=learning_rate)
                        blurred = x

                        def tv_norm(input, tv_beta):
                            img = input
                            row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
                            col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
                            return row_grad + col_grad

                        for _ in range(max_iterations):

                            mask_up = F.upsample(mask.unsqueeze(0).unsqueeze(0), (299, 299), mode='bilinear')
                            # The single channel mask is used with an RGB image,
                            # so the mask is duplicated to have 3 channel,
                            mask_up = mask_up.squeeze(0).repeat(3, 1, 1)
                            # Use the mask to perturbated the input image.
                            perturbated_input = original.mul(1 - mask_up) + \
                                                blurred.mul(mask_up)
                            perturbated_input.data.clamp_(0, 1)

                            outputs = torch.nn.Softmax()(model._model(perturbated_input.unsqueeze(0)))

                            loss = l1_coeff * torch.mean(torch.abs(mask_up)) + \
                                   tv_coeff * tv_norm(mask_up, tv_beta) + outputs[0, category]

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Optional: clamping seems to give better results
                            mask.data.clamp_(0, 1)
                            # #
                            import visdom
                            vis = visdom.Visdom(env='Adversarial Example Showing')
                            vis.images(mask_up, win='mask_')
                            vis.images(original, win='original')
                            vis.images(blurred, win='blurred')
                            vis.images(perturbated_input, win='mask_blurred')

                        if is_adversarial:
                            if return_early:
                                return True
                            else:
                                success = True
                    else:

                        if return_early:
                            return True
                        else:
                            success = True

        return success

    def add_kernel(self,original,kernel,mask):

        original = torch.from_numpy(original).unsqueeze(0).unsqueeze(0).cuda()

        kernel_ = kernel.mul(mask.repeat(1,kernel.size()[1],1,1,1))+self.kernel_gt.mul(1-mask.repeat(1,kernel.size()[1],1,1,1))

        kernel_ = kernel_.div(1e-25+kernel_.sum(dim=1).unsqueeze(1).expand(1,self.kernel_size**2,3,299,299))

        _, denoise = self.kernel_pred(original, kernel_.unsqueeze(0), 1.0)

        return denoise.squeeze(0).detach().cpu().numpy()


    def grad_kernel(self,original, kernel, mask, x_gradient):

        kernel.requires_grad_()
        mask.requires_grad_()

        x_gradient = torch.from_numpy(x_gradient)
        if x_gradient.is_contiguous() is False:
            x_gradient = x_gradient.contiguous()
        x_gradient = x_gradient.unsqueeze(0).cuda()

        original = torch.from_numpy(original).unsqueeze(0).unsqueeze(0).cuda()

        kernel_ = kernel.mul(mask.repeat(1,kernel.size()[1],1,1,1))+self.kernel_gt.mul(1-mask.repeat(1,kernel.size()[1],1,1,1))

        kernel_ = kernel_.div(1e-25+kernel_.sum(dim=1).unsqueeze(1).expand(1,self.kernel_size**2,3,299,299))

        _, denoise = self.kernel_pred(original, kernel_, 1.0)

        loss_fn = torch.nn.L1Loss(reduction='sum')

        l1loss_mask = self.mask_reg*loss_fn(mask,torch.zeros_like(mask))

        denoise.backward(x_gradient)
        l1loss_mask.backward()

        kernel_grad = kernel.grad
        mask_grad = mask.grad

        return kernel_grad,mask_grad

    def adapt_mask(self,original, category, model):

        max_iterations = 150
        tv_beta = 3
        learning_rate = 0.05
        l1_coeff = self.mask_att_l1 #2.0
        tv_coeff = 0.1

        mask_ = torch.zeros([28,28]).cuda()
        mask_.requires_grad_()
        optimizer = torch.optim.Adam([mask_], lr=learning_rate)
        blurred = cv.GaussianBlur(original.transpose(1,2,0), (11, 11), 5).transpose(2,0,1)

        original  = torch.from_numpy(original).cuda()
        blurred = torch.from_numpy(blurred).cuda()

        def tv_norm(input, tv_beta):
            img = input
            row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
            col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
            return row_grad + col_grad

        for i in range(max_iterations):
            mask = F.upsample(mask_, (299, 299), mode='bilinear')
            # The single channel mask is used with an RGB image,
            # so the mask is duplicated to have 3 channel,
            mask = mask.squeeze(0).repeat(3, 1, 1)
            # Use the mask to perturbated the input image.
            perturbated_input = original.mul(1-mask) + \
                                blurred.mul(mask)

            outputs = torch.nn.Softmax()(model._model(perturbated_input.unsqueeze(0)))

            loss = l1_coeff * torch.mean(torch.abs(mask)) + \
                   tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.disp:
                vis = visdom.Visdom(env='Adversarial Example Showing')
                vis.images(mask.data.clamp_(0, 1), win='mask_')

            # Optional: clamping seems to give better results
            mask.data.clamp_(0, 1)

        mask = F.upsample(mask_, (299, 299), mode='bilinear').permute(0,2,3,1)
        mask.data.clamp_(0, 1)
        mask[mask >= 0.5] = 1.
        mask[mask < 0.5] = 0.

        if self.disp:
            vis.images(mask.permute(0,3,1,2), win='mask')

        return mask

    def adapt_mask_gaussblur(self,original, mask_org, category, model):

        max_iterations = 150
        tv_beta = 3
        learning_rate = 0.05
        l1_coeff = self.mask_att_l1 #2.0
        tv_coeff = 0.1
        mask_org = mask_org.squeeze(-1)

        mask_ = F.upsample(mask_org.unsqueeze(0),(28,28),mode='bilinear').detach().cpu()
        mask_ = mask_.cuda()
        mask_.requires_grad_()
        optimizer = torch.optim.Adam([mask_], lr=learning_rate)
        blurred = cv.GaussianBlur(original.transpose(1,2,0), (11, 11), 5).transpose(2,0,1)

        original  = torch.from_numpy(original).cuda()
        blurred = torch.from_numpy(blurred).cuda()

        def tv_norm(input, tv_beta):
            img = input
            row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
            col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
            return row_grad + col_grad

        for i in range(max_iterations):
            mask = F.upsample(mask_, (299, 299), mode='bilinear')
            # The single channel mask is used with an RGB image,
            # so the mask is duplicated to have 3 channel,
            mask = mask.squeeze(0).repeat(3, 1, 1)
            # Use the mask to perturbated the input image.
            perturbated_input = original.mul(1-mask) + \
                                blurred.mul(mask)

            outputs = torch.nn.Softmax()(model._model(perturbated_input.unsqueeze(0)))

            loss = l1_coeff * torch.mean(torch.abs(mask)) + \
                   tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if self.disp:
            #     vis = visdom.Visdom(env='Adversarial Example Showing')
            #     vis.images(mask.data.clamp_(0, 1), win='mask_')

            # Optional: clamping seems to give better results
            mask.data.clamp_(0, 1)

        mask = F.upsample(mask_, (299, 299), mode='bilinear').permute(0,2,3,1)
        mask.data.clamp_(0, 1)
        mask[mask >= 0.5] = 1.
        mask[mask < 0.5] = 0.

        # if self.disp:
        #     vis.images(mask.permute(0,3,1,2), win='mask')

        return mask

    def add_flow_alpha_pert(self,original_, kernel_sz, theta_f,theta_b,alpha,mask):

        original = torch.from_numpy(original_).cuda()
        original = original.unsqueeze(0)

        blur_steps = kernel_sz
        theta_org = torch.tensor([
            [1., 0, 0],
            [0, 1., 0]
        ], dtype=torch.float).cuda()

        seg = mask.permute(0, 3, 1, 2).repeat(blur_steps, 3, 1, 1)
        original = original.repeat(blur_steps, 1, 1, 1)
        theta_f_diff = (theta_f - theta_org) / (blur_steps - 1)
        theta_b_diff = (theta_b - theta_org) / (blur_steps - 1)

        theta_f_diff = theta_f_diff.unsqueeze(0).repeat(blur_steps, 1, 1)
        theta_b_diff = theta_b_diff.unsqueeze(0).repeat(blur_steps, 1, 1)
        theta_range = torch.arange(0, blur_steps).unsqueeze(1).unsqueeze(2).cuda()
        theta_f_ = theta_f_diff * theta_range + theta_org
        theta_b_ = theta_b_diff * theta_range + theta_org

        flow_f = F.affine_grid(theta_f_, original.size())
        flow_b = F.affine_grid(theta_b_, original.size())

        seg_f = F.grid_sample(input=seg, grid=(flow_f), mode='bilinear')
        seg_b = F.grid_sample(input=seg, grid=(flow_b), mode='bilinear')

        seg_f = seg_f.permute(0, 2, 3, 1)[:, :, :, 0].unsqueeze(-1)
        seg_b = seg_b.permute(0, 2, 3, 1)[:, :, :, 0].unsqueeze(-1)

        u_seg_fb = seg_f + seg_b
        u_seg_fb[u_seg_fb > 1] = 1
        u_seg_fb[u_seg_fb < 0] = 0

        flow = flow_f * u_seg_fb + flow_b * (1 - u_seg_fb)

        tensorFlow = flow
        warped = F.grid_sample(input=original, grid=(tensorFlow), mode='bilinear')

        # regularize the warped to make weights out of image domain to be zero
        if self.numSP!=-2:
            min_alpha = alpha.min(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1)
            max_alpha = alpha.max(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1)
            alpha = (alpha - min_alpha) / (max_alpha - min_alpha + 1e-25)
            regbyflow = flow.permute(0,3,1,2)
            for ri in range(2):
                alpha[regbyflow[:, ri, :, :].unsqueeze(1) > 1.] = 0
                alpha[regbyflow[:, ri, :, :].unsqueeze(1) < -1.] = 0
            alpha = alpha / (1e-25+alpha.sum(dim=0))
        else:
            regbyflow = flow.permute(0,3,1,2)
            alpha = alpha.detach()
            for ri in range(2):
                alpha[regbyflow[:, ri, :, :].unsqueeze(1) > 1.] = 0
                alpha[regbyflow[:, ri, :, :].unsqueeze(1) < -1.] = 0

        perted = warped * alpha  # [17,3,h,w]*[17,1,h,w]

        # if self.disp:
        #     vis = visdom.Visdom(env='Adversarial Example Showing')
        #     #vis_flow = flow_vis.flow_to_color(flow.detach().squeeze(0).cpu().numpy(),convert_to_bgr=False)
        #     #plt.imshow(vis_flow)
        #     #vis.matplot(plt, win='flow')
        #     vis.images((warped)[2, :, :, :].squeeze(0).cpu(), win='traned_img2')
        #     vis.images((warped)[4, :, :, :].squeeze(0).cpu(), win='traned_img4')
        #     vis.images((warped)[6,:,:,:].squeeze(0).cpu(), win='traned_img6')
        #     vis.images((warped)[8, :, :, :].squeeze(0).cpu(), win='traned_img8')
        #     vis.images((warped)[10, :, :, :].squeeze(0).cpu(), win='traned_img10')
        #     vis.images((warped)[12,:,:,:].squeeze(0).cpu(), win='traned_img12')
        #     vis.images( warped.mean(dim=0).cpu(),win='mean_img')

        perted  = perted.sum(dim=0)

        return perted.detach().cpu().numpy()

    def grad_flow_alpha_pert(self,original_,kernel_sz,theta_f,theta_b,alpha,mask, x_gradient_):

        original = torch.from_numpy(original_).cuda()
        original = original.unsqueeze(0)

        x_gradient = torch.from_numpy(x_gradient_)
        if x_gradient.is_contiguous() is False:
            x_gradient = x_gradient.contiguous()
        x_gradient = x_gradient.unsqueeze(0)

        blur_steps = kernel_sz

        theta_f.requires_grad_()
        theta_b.requires_grad_()
        alpha.requires_grad_()
        mask.requires_grad_()

        theta_org = torch.tensor([
            [1., 0, 0],
            [0, 1., 0]
        ], dtype=torch.float).cuda()

        seg = mask.permute(0, 3, 1, 2).repeat(blur_steps, 3, 1, 1)

        original = original.repeat(blur_steps, 1, 1, 1)
        theta_f_diff = (theta_f - theta_org) / (blur_steps - 1)
        theta_b_diff = (theta_b - theta_org) / (blur_steps - 1)

        theta_f_diff = theta_f_diff.unsqueeze(0).repeat(blur_steps, 1, 1)
        theta_b_diff = theta_b_diff.unsqueeze(0).repeat(blur_steps, 1, 1)
        theta_range = torch.arange(0, blur_steps).unsqueeze(1).unsqueeze(2).cuda()
        theta_f_ = theta_f_diff * theta_range + theta_org
        theta_b_ = theta_b_diff * theta_range + theta_org

        flow_f = F.affine_grid(theta_f_, original.size())
        flow_b = F.affine_grid(theta_b_, original.size())

        seg_f = F.grid_sample(input=seg, grid=(flow_f), mode='bilinear')
        seg_b = F.grid_sample(input=seg, grid=(flow_b), mode='bilinear')

        seg_f = seg_f.permute(0, 2, 3, 1)[:, :, :, 0].unsqueeze(-1)
        seg_b = seg_b.permute(0, 2, 3, 1)[:, :, :, 0].unsqueeze(-1)

        u_seg_fb = seg_f + seg_b
        u_seg_fb[u_seg_fb > 1] = 1
        u_seg_fb[u_seg_fb < 0] = 0

        flow = flow_f * u_seg_fb + flow_b * (1 - u_seg_fb)

        tensorFlow = flow
        warped = F.grid_sample(input=original, grid=(tensorFlow), mode='bilinear')
        perted = warped * alpha  # [17,3,h,w]*[17,1,h,w]
        perted  = perted.sum(dim=0).unsqueeze(0)

        ## we want to add some constraint to flow or alpha
        self.reg_alpha = False
        self.laplas_alpha = False
        self.gauss_alpha = False

        perted.backward(x_gradient.cuda())

        if self.reg_alpha:
            # regularization for alpha
            alpah_norm = alpha.norm(p=1,dim=0).log()
            alpah_norm.backward(0.05*torch.ones(alpah_norm.size()).cuda())

        if self.laplas_alpha:
            alpha_grad = alpha.grad.squeeze(1).permute(1,2,0).cpu().numpy()
            alpha_grad = cv.Laplacian(alpha_grad, cv.CV_32F, ksize=17)
            alpha_grad = torch.from_numpy(alpha_grad)
            alpha_grad_reg = alpha.grad-1e-6*alpha_grad.permute(2,0,1).unsqueeze(1).cuda()
        else:
            alpha_grad_reg = alpha.grad

        # Gaussian Smooth
        if self.gauss_alpha:
            alpha_grad_reg_ = alpha_grad_reg.squeeze(1).permute(1, 2, 0).cpu().numpy()
            alpha_grad_reg_ = cv.GaussianBlur(alpha_grad_reg_, (7, 7), 100)
            alpha_grad_reg_ = torch.from_numpy(alpha_grad_reg_)
            alpha_grad_reg = alpha_grad_reg_.permute(2, 0, 1).unsqueeze(1).cuda()

        theta_f_grad = theta_f.grad
        theta_b_grad = theta_b.grad

        # Blur attack model attack selection
        if self.blur_model == "whole":
            theta_grad_mean = (theta_f_grad + theta_b_grad)/2
            theta_f_grad = theta_grad_mean
            theta_b_grad = theta_grad_mean
        elif self.blur_model == "obj":
            theta_b_grad = torch.zeros_like(theta_b_grad).cuda()
        elif self.blur_model == "backg":
            theta_f_grad = torch.zeros_like(theta_f_grad).cuda()


        return theta_f_grad,theta_b_grad,alpha_grad_reg,mask.grad

    def init_flow_alpha(self,original,kernel_sz,init_mode="default"):

        h = original.shape[1]
        w = original.shape[2]

        alpha = torch.zeros(kernel_sz,1,h,w)

        alpha[:int(self.alpha_len),:,:,:]=1/self.alpha_len
        print("alpha_len:{}".format(self.alpha_len))
        theta_f,theta_b = self.flow_estimate_saliency(original)

        if init_mode is "rand_alpha":

            alpha_noise = np.random.uniform(
                -0.1, 0.1, alpha.shape).astype(
                alpha.dtype)
            alpha = alpha +alpha_noise
            alpha = self.softmax(alpha)

        alpha = alpha.cuda()

        return theta_f,theta_b, alpha, self.pred

    def flow_estimate_saliency(self,x_):

        # superpixel extraction
        if not hasattr(self, 'superpixel') and self.numSP>0 and self.numSP<600:
            if os.path.exists(self.imgname+"_sp{}.npz".format(self.numSP)):
                r = np.load(self.imgname+"_sp{}.npz".format(self.numSP))
                self.superpixel = r["arr_0"]
                self.sp_tensor = torch.from_numpy(r["arr_1"]).cuda()
                self.sp_pixelnum = torch.from_numpy(r["arr_2"]).cuda()
            else:
                from skimage.segmentation import slic
                x_ = x_.transpose(1, 2, 0).astype(np.double)
                self.superpixel = slic(x_, n_segments=self.numSP, sigma=5)

                realSPnum = self.superpixel.max()
                sp = torch.from_numpy(self.superpixel).cuda()
                self.sp_tensor = torch.zeros(realSPnum + 1, sp.size()[0], sp.size()[1]).cuda()
                for spi in range(realSPnum + 1):
                    self.sp_tensor[spi, :, :][sp == spi] = 1.

                self.sp_pixelnum = self.sp_tensor.sum(dim=1).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                self.sp_tensor = self.sp_tensor.unsqueeze(1).unsqueeze(1).repeat(1, self.kernel_size, 1, 1, 1)

                np.savez(self.imgname+"_sp{}".format(self.numSP),self.superpixel,
                         self.sp_tensor.detach().cpu(),self.sp_pixelnum.detach().cpu())


        if not hasattr(self, 'pred') and not os.path.exists(self.imgname):

            if not hasattr(self, 'net_saliecny'):
                self.net_saliecny = build_model("resnet").cuda()
                self.net_saliecny.eval()  # use_global_stats = True
                self.net_saliecny.apply(weights_init)
                self.net_saliecny.base.load_pretrained_model(
                    torch.load("./tools/saliency/dataset/pretrained/resnet50_caffe.pth"))
                self.net_saliecny.load_state_dict(torch.load("./tools/saliency/results/run-0/models/final.pth"))
                self.net_saliecny.eval()  # use_global_stats = True
                net_saliecny = self.net_saliecny
            else:
                net_saliecny = self.net_saliecny

            # forward pass
            x_ =x_*255
            x_ =x_- np.array((104.00699, 116.66877, 122.67892))[:,np.newaxis,np.newaxis]
            x = torch.Tensor(x_).cuda()
            x = x.unsqueeze(0)

            pred_ = net_saliecny(x)
            pred_ = torch.sigmoid(pred_)
            pred = pred_
            pred[pred_>3e-1]=1
            pred[pred_<=3e-1] = 0

            # sementic segmentation regularized flow
            self.pred = pred.permute(0, 2, 3, 1).detach()

            if self.disp:
                vis = visdom.Visdom(env='Adversarial Example Showing')
                vis.images(pred_.squeeze(0).repeat(3,1,1).float(), win='saliency_res')

            import imageio
            save_pred = pred_ * 255
            save_pred = save_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
            imageio.imwrite(self.imgname, save_pred)

            print("saliency detection succeeds!")

        elif not hasattr(self, 'pred') and os.path.exists(self.imgname):
            import imageio
            pred_org = torch.from_numpy(imageio.imread(self.imgname))/255.
            pred_ = pred_org.unsqueeze(0).unsqueeze(-1)
            pred = pred_
            pred[pred_>3e-1]=1
            pred[pred_<=3e-1] = 0
            self.pred = pred.float().cuda()

            if self.disp:
                vis = visdom.Visdom(env='Adversarial Example Showing')
                vis.images(pred_org.unsqueeze(0).repeat(3,1,1).float()*torch.from_numpy(x_).unsqueeze(0), win='saliency_res')

            print("saliency detection loaded!")

        if self.numSP ==-2 or self.blur_model=="joint_wo_adaptmot":
            theta_b = torch.tensor([
                [1.0, 0., self.trans_val],
                [0., 1.0, self.trans_val]
            ], dtype=torch.float).cuda()

            theta_f = torch.tensor([
                [1.0, 0., self.trans_val],
                [0., 1.0, self.trans_val]
            ], dtype=torch.float).cuda()
            print("numsp:{}, blurmodel:{}, trans_val:{}".format(self.numSP,self.blur_model,self.trans_val))
        else:
            theta_b = torch.tensor([
                [1.0, 0., 0.1],
                [0., 1.0, 0.]
            ], dtype=torch.float).cuda()

            theta_f = torch.tensor([
                [1.0, 0., 0.1],
                [0., 1.0, 0.]
            ], dtype=torch.float).cuda()

        return theta_f,theta_b


class GDOptimizerMixin(object):
    def _create_optimizer(self, a, stepsize):
        return GDOptimizer(stepsize)


class AdamOptimizerMixin(object):
    def _create_optimizer(self, a, stepsize):
        return AdamOptimizer(a.unperturbed.shape, a.unperturbed.dtype, stepsize)


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class SparseL1GradientMixin(object):
    """Calculates a sparse L1 gradient introduced in [1]_.

         References
        ----------
        .. [1] Florian Tramèr, Dan Boneh,
               "Adversarial Training and Robustness for Multiple Perturbations",
               https://arxiv.org/abs/1904.13000

        """

    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        q = gradient_args["q"]

        gradient = yield from a.gradient_one(x, class_, strict=strict)

        # make gradient sparse
        abs_grad = np.abs(gradient)
        gradient_percentile_mask = abs_grad <= np.percentile(abs_grad.flatten(), q)
        e = np.sign(gradient)
        e[gradient_percentile_mask] = 0

        # using mean to make range of epsilons comparable to Linf
        normalization = np.mean(np.abs(e))

        gradient = e / normalization
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient_norm = np.mean(np.abs(gradient))
        gradient_norm = max(1e-12, gradient_norm)
        gradient = gradient / gradient_norm
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient_norm = max(1e-12, gradient_norm)
        gradient = gradient / gradient_norm
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning(
                "Running an attack that tries to minimize the"
                " Linfinity norm of the perturbation without"
                " specifying foolbox.distances.Linfinity as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning(
                "Running an attack that tries to minimize the"
                " L1 norm of the perturbation without"
                " specifying foolbox.distances.MAE as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning(
                "Running an attack that tries to minimize the"
                " L2 norm of the perturbation without"
                " specifying foolbox.distances.MSE as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class LinfinityBasicIterativeAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Basic Iterative Method introduced in [1]_.

    This attack is also known as Projected Gradient
    Descent (PGD) (without random start) or FGMS^k.

    References
    ----------
    .. [1] Alexey Kurakin, Ian Goodfellow, Samy Bengio,
           "Adversarial examples in the physical world",
            https://arxiv.org/abs/1607.02533

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
        pert_type = "Add",
        blur_model = "blur_model",
        numSP = -1,
        mask_att_l1 = 2.0,
        direction = None,
        imgname = None
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        if isinstance(epsilon,np.ndarray):
            for x in epsilon.tolist():
                assert x>0
        else:
            assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early, pert_type=pert_type,\
            blur_model = blur_model,numSP=numSP,mask_att_l1 = mask_att_l1,direction = direction,imgname = imgname
        )

BasicIterativeMethod = LinfinityBasicIterativeAttack
BIM = BasicIterativeMethod


class L1BasicIterativeAttack(
    L1GradientMixin,
    L1ClippingMixin,
    L1DistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class SparseL1BasicIterativeAttack(
    SparseL1GradientMixin,
    L1ClippingMixin,
    L1DistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Sparse version of the Basic Iterative Method
    that minimizes the L1 distance introduced in [1]_.

     References
    ----------
    .. [1] Florian Tramèr, Dan Boneh,
           "Adversarial Training and Robustness for Multiple Perturbations",
           https://arxiv.org/abs/1904.13000

    .. seealso:: :class:`L1BasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        q=80.0,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Sparse version of a gradient-based attack that minimizes the
        L1 distance.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        q : float
            Relative percentile to make gradients sparse (must be in [0, 100))
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        assert 0 <= q < 100.0, "`q` must be in [0, 100)."

        yield from self._run(
            a,
            binary_search,
            epsilon,
            stepsize,
            iterations,
            random_start,
            return_early,
            gradient_args={"q": q},
        )


class L2BasicIterativeAttack(
    L2GradientMixin,
    L2ClippingMixin,
    L2DistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class ProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_ without random start.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


ProjectedGradientDescent = ProjectedGradientDescentAttack
PGD = ProjectedGradientDescent


class RandomStartProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_ with random start.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=True,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


RandomProjectedGradientDescent = RandomStartProjectedGradientDescentAttack
RandomPGD = RandomProjectedGradientDescent


class MomentumIterativeAttack(
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Momentum Iterative Method attack
    introduced in [1]_. It's like the Basic
    Iterative Method or Projected Gradient
    Descent except that it uses momentum.

    References
    ----------
    .. [1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su,
           Jun Zhu, Xiaolin Hu, Jianguo Li, "Boosting Adversarial
           Attacks with Momentum",
           https://arxiv.org/abs/1710.06081

    """

    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        # get current gradient
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        success = yield from super(MomentumIterativeAttack, self)._run_one(
            *args, **kwargs
        )
        return success

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.06,
        iterations=10,
        decay_factor=1.0,
        random_start=False,
        return_early=True,
        pert_type="Add",
        blur_model="blur_model",
        numSP = -1,
        mask_att_l1 = 2.0,
        direction = None,
        imgname=None
    ):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        if isinstance(epsilon,np.ndarray):
            for x in epsilon.tolist():
                assert x>=0
        else:
            assert epsilon > 0

        self._decay_factor = decay_factor

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early, pert_type=pert_type,\
            blur_model = blur_model,numSP=numSP,mask_att_l1=mask_att_l1,direction = direction,imgname = imgname
        )


MomentumIterativeMethod = MomentumIterativeAttack


class AdamL1BasicIterativeAttack(
    L1GradientMixin,
    L1ClippingMixin,
    L1DistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance using the Adam optimizer.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class AdamL2BasicIterativeAttack(
    L2GradientMixin,
    L2ClippingMixin,
    L2DistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance using the Adam optimizer.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class AdamProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_, [2]_ without random start using the Adam optimizer.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. [2] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


AdamProjectedGradientDescent = AdamProjectedGradientDescentAttack
AdamPGD = AdamProjectedGradientDescent


class AdamRandomStartProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_, [2]_ with random start using the Adam optimizer.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. [2] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=True,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


AdamRandomProjectedGradientDescent = (
    AdamRandomStartProjectedGradientDescentAttack  # noqa: E501
)

AdamRandomPGD = AdamRandomProjectedGradientDescent
