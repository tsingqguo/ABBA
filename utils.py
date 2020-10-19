import yaml
import os
import numpy as np
import imageio

def _load_image(path):
    """
        Reads image image from the given path and returns an numpy array.
    """
    image = np.load(path)
    assert image.dtype == np.uint8
    assert image.shape == (64, 64, 3)
    return image


def _read_image(file_name):
    """
        Returns a tuple of image as numpy array and label as int,
        given the csv row.
    """
    input_folder = "test_images/"
    img_path = os.path.join(input_folder, file_name)
    image = _load_image(img_path)
    assert image.dtype == np.uint8
    image = image.astype(np.float32)
    assert image.dtype == np.float32
    return image


def read_images():
    """
        Returns a list containing tuples of images as numpy arrays
        and the correspoding label.
        In case of an untargeted attack the label is the ground truth label.
        In case of a targeted attack the label is the target label.
    """
    filepath = "test_images/labels.yml"
    with open(filepath, 'r') as ymlfile:
        data = yaml.load(ymlfile)

    data_key = list(data.keys())
    data_key.sort()
    return [(key, _read_image(key), data[key]) for key in data_key]


def check_image(image):
    # image should a 64 x 64 x 3 RGB image
    assert(isinstance(image, np.ndarray))

    if len(image.shape)>3:
        image = image.squeeze(0).transpose(1,2,0)

    assert(image.shape == (64, 64, 3) or image.shape == (224, 224, 3) or image.shape == (299, 299, 3) or image.shape == (28, 28, 1))

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


def store_adversarial(file_name, original, adversarial):
    """
        Given the filename, stores the adversarial as .npy file.
    """
    result_root = "/mnt/nvme/projects/BlurAttack/"
    #result_root = "/home/wangjian/tsingqguo/BlurAttack/"

    path = os.path.join(result_root+"results", file_name)
    path_without_extension = os.path.splitext(path)[0]
    np.save(path_without_extension, adversarial)
    #np.save(path_without_extension + "_org", original)

    original = check_image(original)
    if adversarial is not None:
        adversarial = check_image(adversarial)
    #from scipy import
    import imageio
    imageio.imwrite(path_without_extension+".jpg", adversarial)
    imageio.imwrite(path_without_extension + "_org.jpg", original)
    print("Saving result:" + file_name)

def save_adversarial(path, adversarial):
    """
        Given the filename, stores the adversarial as .png file.
    """
    path_without_extension = os.path.splitext(path)[0]
    np.save(path_without_extension, adversarial)

    if adversarial is not None:
        adversarial = check_image(adversarial)

    imageio.imwrite(path_without_extension+".png", adversarial)
    print("Saving result:" + path)

def load_adversarial(file_name,images):
    """
        Given the filename, stores the adversarial as .npy file.
    """
    result_root = "/mnt/nvme/projects/BlurAttack/" #"/home/wangjian/tsingqguo/BlurAttack/" #

    path = os.path.join(result_root+"results", file_name)
    path_without_extension = path#os.path.splitext(path)[0]
    adversarial = np.load(path_without_extension+".npy")
    if len(adversarial.shape)==3:
        adversarial = adversarial.transpose(2,0,1)[np.newaxis]
    if adversarial.max()>1:
        adversarial = adversarial.astype(np.float32)/255

    if os.path.exists(path_without_extension + "_org.npy"):
        original = np.load(path_without_extension + "_org.npy")
    else:
        original = images.cpu().detach().numpy()#.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    if len(original.shape)==3:
        original = original.transpose(2,0,1)[np.newaxis]
    if original.max()>1:
        original = original.astype(np.float32)/255

    if adversarial.sum()!=0 and not np.isnan(adversarial.max()):
        diff = np.linalg.norm(adversarial-original)
        # status : 0 org has already been adversarial
        # :1 adversarial example has been found
        # :-1 adversarial example is not found
        if diff==0:
            status = 0
        elif diff>0:
            status = 1
        elif np.isnan(diff.max()):
            status = -1
    else:
        status = -1

    return status, original, adversarial

def compute_MAD(dataset):

    result_root = "/mnt/nvme/projects/BlurAttack/"


    def load_image(path):
        x = np.load(path)
        assert (x.shape == (64, 64, 3) or x.shape == (224, 224, 3))
        assert x.dtype == np.uint8
        return x
    def distance(X, Y):
        assert X.dtype == np.uint8
        assert Y.dtype == np.uint8
        X = X.astype(np.float64) / 255
        Y = Y.astype(np.float64) / 255
        return np.linalg.norm(X - Y)
    # distance if no adversarial was found (worst case)
    def worst_case_distance(X):
        assert X.dtype == np.uint8
        worst_case = np.zeros_like(X)
        worst_case[X < 128] = 255
        return distance(X, worst_case)

    distances = []
    real_distances = []

    path = os.path.join(result_root+"/results", dataset)

    for file in os.listdir(path):
        if os.path.splitext(file)[1] == ".jpg":
            continue
        filename = os.path.splitext(file)[0]
        if filename[-4:] == "_org":
            continue

        original = load_image(os.path.join(result_root+"results", dataset,'{}_org.npy'.format(filename)))
        try:
            adversarial = load_image(os.path.join(result_root+"results", dataset,'{}.npy'.format(filename)))
        except AssertionError:
            #print('adversarial for {} is invalid'.format(file))
            adversarial = None
        if adversarial is None:
            _distance = float(worst_case_distance(original))
        else:
            _distance = float(distance(original, adversarial))
        real_distances.append(_distance)

    real_distances = np.array(real_distances)
    distances = real_distances * 255

    print("\tMedian Distance:  %.6f"  %np.median(real_distances[distances > 50]))
    print("\tMean Distance:    %.6f"  %np.mean(real_distances[distances > 50]))