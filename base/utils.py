import json
import h5py
import numpy as np
import sys
import os
from PIL import Image, ImageDraw
import cv2
import collections
import cffi

IMG_SCALE = (64, 64)

def load_h5py(filepath):
    f = h5py.File(filepath, "r")
    shape = f["shape"]
    matrix = f["matrix"]
    img = np.array(matrix)
    img = img.reshape(shape[0], shape[1])
    f.close()
    if np.any(np.isnan(img)):
        return None
    else:
        return img

def write_h5py(img, path):
    img_copy = img.copy()
    h5f = h5py.File(path, 'w')
    shape = h5f.require_dataset("shape", (2,), dtype='int')
    # shape.resize(1, axis=0)
    shape[0] = img_copy.shape[0]
    shape[1] = img_copy.shape[1]
    shape = img.shape
    matrix = h5f.require_dataset("matrix", (img.size, ), dtype='float')
    img_flaten = img.flatten()
    for i in range(img_flaten.shape[0]):
        matrix[i] = img_flaten[i]
    h5f.close()

def depth_to_normal(img_d):
    img_arr = img_d.copy()
    ffi = cffi.FFI()
    ffi.cdef("bool NormalMapFromPointsSetPython(float *image_depth, unsigned char *image_normal, float *mask, int width, int height);")
    c = ffi.dlopen("./normalmap.so")
    depth_map = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    normal_map = ffi.new("unsigned char[{}]".format(img_arr.shape[0] * img_arr.shape[1]*3), b"\0")
    mask = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    c.NormalMapFromPointsSetPython(depth_map, normal_map, mask, img_arr.shape[1], img_arr.shape[0])
    new_normal_map = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype="uint8")
    for j in range(img_arr.shape[0]):
        for k in range(img_arr.shape[1]):
            for i in range(3):
                new_normal_map[j][k][2-i] = normal_map[j * img_arr.shape[1] * 3 + k * 3 + i]
    new_normal_map = np.array(Image.fromarray(new_normal_map).resize(IMG_SCALE, resample=Image.BILINEAR))
    return new_normal_map