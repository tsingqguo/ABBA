from foolbox.models import PyTorchModel
from pytorchcv.model_provider import get_model as ptcv_get_model

def create_fmodel(dataset="tiny_imagenet",model_name="resnet18",gpu=None):

    if dataset == "imagenet":

        model = ptcv_get_model(model_name, pretrained=True)
        model.eval()
        if gpu is not None:
            model = model.cuda()
        #
        # def preprocessing(x):
        #     mean = np.array([0.485, 0.456, 0.406])
        #     std = np.array([0.229, 0.224, 0.225])
        #     _mean = mean.astype(x.dtype)
        #     _std = std.astype(x.dtype)
        #     x = x - _mean
        #     x /= _std
        #
        #     assert x.ndim in [3, 4]
        #     if x.ndim == 3:
        #         x = np.transpose(x, axes=(2, 0, 1))
        #     elif x.ndim == 4:
        #         x = np.transpose(x, axes=(0, 3, 1, 2))
        #
        #     def grad(dmdp):
        #         assert dmdp.ndim == 3
        #         dmdx = np.transpose(dmdp, axes=(1, 2, 0))
        #         return dmdx / _std
        #     return x, grad

        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

        fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

    elif dataset == "cifa10":

        model = ptcv_get_model(model_name, pretrained=True)
        model.eval()
        if gpu is not None:
            model = model.cuda()

        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)

        fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preprocessing)

    elif dataset == "dev":

        model = ptcv_get_model(model_name, pretrained=True)
        model.eval()
        if gpu is not None:
            model = model.cuda()

        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)

        fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

    return fmodel

