import numpy as np


def tensor_to_numpy(tensor, squeze_first=False):
    if "tensor" in str(type(tensor)):
        array = tensor.numpy()
    else:
        array = tensor
    if squeze_first and (array.shape[0] == 1):
        array = np.squeeze(array, axis=0)
    return array


def squeeze_array(array, keep_first=True):
    if keep_first:
        axis = tuple(np.where(np.array(array.shape)[1:] == 1)[0] + 1)
        array = np.squeeze(array, axis=axis)
    else:
        array = np.squeeze(array)
    return array
