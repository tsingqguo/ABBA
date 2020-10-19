import numpy as np
from collections import Iterable
import logging
import abc

from .base import Attack
from .base import generator_decorator

from tools.saliency.networks.poolnet import build_model, weights_init
import torch
import os
from torch.nn import functional as F

class SingleStepGradientBaseAttack(Attack):
    """Common base class for single step gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a):
        raise NotImplementedError

    def _run(self, a, epsilons, max_epsilon,type="whole",imgname=None):
        if not a.has_gradient():
            return
        x = a.unperturbed
        min_, max_ = a.bounds()

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
            self.pred = pred.squeeze(0).detach().cpu().nump()

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

        gradient = yield from self._gradient(a)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, max_epsilon, num=epsilons + 1)[1:]
            decrease_if_first = True
        else:
            decrease_if_first = False

        for _ in range(2):  # to repeat with decreased epsilons if necessary
            for i, epsilon in enumerate(epsilons):

                perturbed = x + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                if type == "obj":
                    perturbed = (1 - self.pred) * x + self.pred * perturbed
                elif type == "bg":
                    perturbed = (1 - self.pred) * perturbed + self.pred * x

                _, is_adversarial = yield from a.forward_one(perturbed)

                if is_adversarial:
                    if type == "att":

                        max_iterations = 150
                        model = a._model
                        category = a.original_class
                        tv_beta = 3
                        learning_rate = 0.05
                        l1_coeff = 0.05
                        tv_coeff = 0.1
                        mask = torch.zeros([28, 28]).cuda()

                        mask.requires_grad_()
                        optimizer = torch.optim.Adam([mask], lr=learning_rate)

                        original = torch.from_numpy(x).cuda()
                        perturbed = torch.from_numpy(perturbed).cuda()

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
                                                perturbed.mul(mask_up)
                            perturbated_input.data.clamp_(0, 1)

                            outputs = torch.nn.Softmax()(model._model(perturbated_input.unsqueeze(0)))

                            loss = l1_coeff * torch.mean(torch.abs(mask_up)) + \
                                   tv_coeff * tv_norm(mask_up, tv_beta) + outputs[0, category]

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Optional: clamping seems to give better results
                            mask.data.clamp_(0, 1)
                            #
                            # import visdom
                            # vis = visdom.Visdom(env='Adversarial Example Showing')
                            # vis.images(mask_up, win='mask_')
                            # vis.images(original, win='original')
                            # vis.images(perturbed, win='perturbed')
                            # vis.images(perturbated_input, win='mask_perturbed')

                            _, is_adversarial = yield from a.forward_one(perturbated_input.cpu().detach().numpy())

                            if is_adversarial:
                                if decrease_if_first and i < 20:
                                    logging.info("repeating attack with smaller epsilons")
                                    break
                                return
                    else:
                        if decrease_if_first and i < 20:
                            logging.info("repeating attack with smaller epsilons")
                            break
                        return


            max_epsilon = epsilons[i]
            epsilons = np.linspace(0, max_epsilon, num=20 + 1)[1:]


class GradientAttack(SingleStepGradientBaseAttack):
    """Perturbs the input with the gradient of the loss w.r.t. the input,
    gradually increasing the magnitude until the input is misclassified.

    Does not do anything if the model does not have a gradient.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=1000, max_epsilon=1):
        """Perturbs the input with the gradient of the loss w.r.t. the input,
        gradually increasing the magnitude until the input is misclassified.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the gradient direction
            or number of step sizes between 0 and max_epsilon that should
            be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = yield from a.gradient_one()
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient = gradient / (gradient_norm + 1e-8) * (max_ - min_)
        return gradient


class GradientSignAttack(SingleStepGradientBaseAttack):
    """Adds the sign of the gradient to the input, gradually increasing
    the magnitude until the input is misclassified. This attack is
    often referred to as Fast Gradient Sign Method and was introduced
    in [1]_.

    Does not do anything if the model does not have a gradient.

    References
    ----------
    .. [1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy,
           "Explaining and Harnessing Adversarial Examples",
           https://arxiv.org/abs/1412.6572
    """

    @generator_decorator
    def as_generator(self, a, epsilons=1000, max_epsilon=1,type="whole",imgname=None):
        """Adds the sign of the gradient to the input, gradually increasing
        the magnitude until the input is misclassified.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon,type=type,imgname=imgname)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = yield from a.gradient_one()
        gradient = np.sign(gradient) * (max_ - min_)
        return gradient


FGSM = GradientSignAttack
