import numpy as np
import os
import torch
from collections import Iterable
from torch.nn import functional as F
import cv2 as cv
from pyblur import LinearMotionBlur,DefocusBlur

from scipy.ndimage.filters import gaussian_filter

from .base import Attack
from .base import generator_decorator

from tools.saliency.networks.poolnet import build_model, weights_init


class GaussianBlurAttack(Attack):
    """Blurs the input until it is misclassified."""

    @generator_decorator
    def as_generator(self, a, epsilons=1000,regiontype="whole",blurtype = "gaussian",imgname=None):

        """Blurs the input until it is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if input is a `numpy.ndarray`, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int or Iterable[float]
            Either Iterable of standard deviations of the Gaussian blur
            or number of standard deviations between 0 and 1 that should
            be tried.

        """

        x = a.unperturbed
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        hw = [x.shape[i] for i in range(x.ndim) if i != axis]
        h, w = hw
        size = max(h, w)

        self.imgname = imgname

        if not hasattr(self, 'pred') and self.imgname is not None and not os.path.exists(self.imgname):

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
            x_ =x*255
            x_ =x_- np.array((104.00699, 116.66877, 122.67892))[:,np.newaxis,np.newaxis]
            x = torch.Tensor(x_).cuda()
            x = x.unsqueeze(0)

            pred_ = net_saliecny(x)
            pred_ = torch.sigmoid(pred_)
            pred = pred_
            pred[pred_>3e-1]=1
            pred[pred_<=3e-1] = 0

            # sementic segmentation regularized flow
            self.pred = pred.squeeze(0).detach().cpu().numpy()

            import imageio
            save_pred = pred_ * 255
            save_pred = save_pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
            imageio.imwrite(self.imgname, save_pred)

        elif not hasattr(self, 'pred') and os.path.exists(self.imgname):

            import imageio
            pred_org = imageio.imread(self.imgname)/ 255.
            pred_ = pred_org[np.newaxis]
            pred = pred_
            pred[pred_ > 3e-1] = 1
            pred[pred_ <= 3e-1] = 0
            self.pred = pred.astype(np.float32)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            # epsilon = 1 will correspond to
            # sigma = size = max(width, height)
            sigmas = [epsilon * size] * 3
            sigmas[axis] = 0

            # print("blurtype:{}".format(blurtype))
            if blurtype == "gaussian":
                blurred = gaussian_filter(x, sigmas)
            elif blurtype == "motionblur":
                blurred = LinearMotionBlur(x, int(epsilon), 45, 'full')
            elif blurtype == "defocublur":
                blurred = DefocusBlur(x, int(epsilon))

            blurred = np.clip(blurred, min_, max_)
            if regiontype == "obj":
                blurred = (1-self.pred)*x + self.pred*blurred
            elif regiontype == "bg":
                blurred = (1-self.pred)*blurred+ self.pred*x
            _, is_adversarial = yield from a.forward_one(blurred)

            if is_adversarial:
                if regiontype == "att":

                    max_iterations = 150
                    model = a._model
                    category = a.original_class
                    tv_beta = 3
                    learning_rate = 0.05
                    l1_coeff = 0.05#0.05
                    tv_coeff = 0.1
                    mask = torch.zeros([28, 28]).cuda()

                    mask.requires_grad_()
                    optimizer = torch.optim.Adam([mask], lr=learning_rate)

                    original = torch.from_numpy(x).cuda()
                    blurred = torch.from_numpy(blurred).cuda()

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
                        # import visdom
                        # vis = visdom.Visdom(env='Adversarial Example Showing')
                        # vis.images(mask_up, win='mask_')
                        # vis.images(original, win='original')
                        # vis.images(blurred, win='blurred')
                        # vis.images(perturbated_input, win='mask_blurred')

                        _, is_adversarial = yield from a.forward_one(perturbated_input.cpu().detach().numpy())

                        if is_adversarial:
                            return
                else:
                    return

