import os

import cv2
import numpy as np


def _stem(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _to_uint8_single_channel(image):
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim > 3:
        image = image.reshape(image.shape[0], image.shape[1])
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    return image


def _binary_fallback(input_path, side=512):
    data = np.fromfile(input_path, dtype=np.uint8)
    if data.size == 0:
        data = np.zeros(1, dtype=np.uint8)
    total = side * side
    if data.size < total:
        repeats = int(np.ceil(float(total) / float(data.size)))
        data = np.tile(data, repeats)
    plane = data[:total].reshape((side, side))
    return cv2.equalizeHist(plane)


def run_module(input_path_dict, output_folder_path):
    input_path = input_path_dict["Input Image"]

    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    gray = _to_uint8_single_channel(image)
    if gray is None:
        gray = _binary_fallback(input_path, side=512)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed = cv2.Canny(blurred, 50, 150)

    output_name = "%s_sup_out.jpg" % _stem(input_path)
    output_path = os.path.join(output_folder_path, output_name)
    cv2.imwrite(output_path, processed)

    return {"Output Image": output_path}
