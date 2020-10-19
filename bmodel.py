from foolbox.models import PyTorchModel
from pytorchcv.model_provider import get_model as ptcv_get_model

def create_bmodel(dataset="tiny_imagenet",model_name="resnet101",gpu=None,params=None):

    if dataset == "imagenet":

        model = ptcv_get_model(model_name, pretrained=True)
        model.eval()
        if gpu is not None:
            model = model.cuda()

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
        #
        #     return x, grad

        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

        bmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

    elif dataset == "cifa10":

        model = ptcv_get_model(model_name, pretrained=True)
        model.eval()
        if gpu is not None:
            model = model.cuda()

        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)

        bmodel = PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preprocessing)

    elif dataset in ["dev","sharp","real"]:

        model = ptcv_get_model(model_name, pretrained=True)

        model.eval()

        if gpu is not None:
            model = model.cuda()

        preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)

        bmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

    elif dataset == "mnist":

        import tools.spatial_transformer.model as stn_model
        from tools.spatial_transformer.model import initialize
        from tools.spatial_transformer.vision_transforms import gen_random_perspective_transform, apply_transform_to_batch
        from tools.spatial_transformer import utils as stn_utils

        P_init = gen_random_perspective_transform(params)

        model = stn_model.STN(getattr(stn_model, params.stn_module), params, P_init).to(params.device)
        initialize(model)
        stn_utils.load_checkpoint('./tools/spatial_transformer/experiments/base_stn_model/state_checkpoint.pt', model)

        bmodel = PyTorchModel(model, bounds=(0, 1), num_classes=10)



    return bmodel

