import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import os
import re
import imageio

IMAGE_SZ = 128  

def load_city_image():
    im = Image.open('images/city_128.png').convert('RGB')
    width, height = im.size
    left = (width - IMAGE_SZ) / 2
    top = (height - IMAGE_SZ) / 2
    im = im.crop((left, top, left + IMAGE_SZ, top + IMAGE_SZ))
    pix = np.array(im)
    assert pix.shape == (IMAGE_SZ, IMAGE_SZ, 3)
    return pix[np.newaxis] / 255.0 # Need to normalize images to [0, 1]

# Loads multiple images from a directory.
# Returns: normalized numpy array of size (m, IMAGE_SZ, IMAGE_SZ, 3)
def load_images(in_PATH, verbose=False):
    imgs = []
    for filename in sorted(os.listdir(in_PATH)):
        if verbose:
            print('Processing %s' % filename)
        full_filename = os.path.join(os.path.abspath(in_PATH), filename)
        img = Image.open(full_filename).convert('RGB')
        pix = np.array(img)
        pix_norm = pix / 255.0
        imgs.append(pix_norm)
    return np.array(imgs)

# Reads in all the images in a directory and saves them to an .npy file.
def compile_images(in_PATH, out_PATH):
    imgs = load_images(in_PATH, verbose=True)
    np.save(out_PATH, imgs)

# Masks and preprocesses an (m, IMAGE_SZ, IMAGE_SZ, 3) batch of images for image outpainting.
# Returns: numpy array of size (m, IMAGE_SZ, IMAGE_SZ, 4)
def preprocess_images_outpainting(imgs, crop=True):
    m = imgs.shape[0]
    imgs = np.array(imgs, copy=True)
    pix_avg = np.mean(imgs, axis=(1, 2, 3))
    if crop:
        imgs[:, :, :int(2 * IMAGE_SZ / 8), :] = imgs[:, :, int(-2 * IMAGE_SZ / 8):, :] = pix_avg[:, np.newaxis, np.newaxis, np.newaxis]
    mask = np.zeros((m, IMAGE_SZ, IMAGE_SZ, 1))
    mask[:, :, :int(2 * IMAGE_SZ / 8), :] = mask[:, :, int(-2 * IMAGE_SZ / 8):, :] = 1.0
    imgs_p = np.concatenate((imgs, mask), axis=3)
    return imgs_p

# Expands and preprocesses a single (h, w, 3) image for image outpainting.
# Returns: numpy array of size (h, w + 2 * dw, 4)
def preprocess_images_gen(img):
    img = np.array(img, copy=True)
    pix_avg = np.mean(img)
    dw = int(2 * IMAGE_SZ / 8) # Amount that will be outpainted on each side
    img_expand = np.ones((img.shape[0], img.shape[1] + 2 * dw, img.shape[2])) * pix_avg
    img_expand[:, dw:-dw, :] = img
    mask = np.zeros((img_expand.shape[0], img_expand.shape[1], 1))
    mask[:, :int(2 * IMAGE_SZ / 8), :] = mask[:, int(-2 * IMAGE_SZ / 8):, :] = 1.0
    img_p = np.concatenate((img_expand, mask), axis=2)
    return img_p[np.newaxis]

# Renormalizes an image to [0, 255].
def norm_image(img_r):
    img_norm = (img_r * 255.0).astype(np.uint8)
    return img_norm

# Visualize an image.
def vis_image(img_r, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.show()

# Save an image as a .png file.
def save_image(img_r, name, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img= img.convert("RGB")
    img.save(name, format='PNG')

# Sample a random minibatch from data.
# Returns: Two numpy arrays, representing examples and their corresponding
#          preprocessed arrays.
def sample_random_minibatch(data, data_p, m):
    indices = np.random.randint(0, data.shape[0], m)
    return data[indices], data_p[indices]

# Plots the loss and saves the plot.
def plot_loss(loss_filename, title, out_filename):
    loss = np.load(loss_filename)
    assert 'train_loss' in loss and 'out_loss' in loss
    train_loss = loss['train_loss']
    out_loss = loss['out_loss'] # TODO: Deal with out_loss not changing during Phase 2
    label_train, = plt.plot(train_loss[:, 0], train_loss[:, 1], label='Training loss')
    label_out, = plt.plot(out_loss[:, 0], out_loss[:, 1], label='OUTPUT loss')
    plt.legend(handles=[label_train, label_out])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(out_filename)
    plt.show()
    plt.clf()

# Use seamless cloning to improve the generator's output.
def postprocess_images_outpainting(img_PATH, img_o_PATH, out_PATH, blend=False): # img, img_0 are (64, 64, 3), mask is (64, 64, 1)
    src = cv2.imread(img_PATH)[:, int(2 * IMAGE_SZ / 8):-int(2 * IMAGE_SZ / 8), :]
    dst = cv2.imread(img_o_PATH)
    if blend:
        mask = np.ones(src.shape, src.dtype) * 255
        center = (int(IMAGE_SZ / 2) - 1, int(IMAGE_SZ / 2) - 1)
        out = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    else:
        out = dst.copy()
        out[:, int(2 * IMAGE_SZ / 8):-int(2 * IMAGE_SZ / 8), :] = src
    cv2.imwrite(out_PATH, out)

# Use seamless cloning to improve the generator's output.
def postprocess_images_gen(img, img_o, blend=False):
    src = img[:, :, ::-1].copy()
    dst = img_o[:, :, ::-1].copy()
    if blend:
        mask = np.ones(src.shape, src.dtype) * 255
        center = (int(dst.shape[1] / 2) - 1, int(dst.shape[0] / 2) - 1)
        out = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    else:
        out = dst.copy()
        out[:, int(2 * IMAGE_SZ / 8):-int(2 * IMAGE_SZ / 8), :] = src
    return out[:, :, ::-1].copy()

# Crop and resize all the images in a directory.
def resize_images(src_PATH, dst_PATH):
    for filename in os.listdir(src_PATH):
        print('Processing %s' % filename)
        full_filename = os.path.join(os.path.abspath(src_PATH), filename)
        img_raw = Image.open(full_filename).convert('RGB')
        w, h = img_raw.size
        if w <= h:
            dim = w
            y_start = int((h - dim) / 2)
            img_crop = img_raw.crop(box=(0, y_start, dim, y_start + dim))
        else: # w > h
            dim = h
            x_start = int((w - dim) / 2)
            img_crop = img_raw.crop(box=(x_start, 0, x_start + dim, dim))
        img_scale = img_crop.resize((IMAGE_SZ, IMAGE_SZ), Image.ANTIALIAS)
        full_outfilename = os.path.join(os.path.abspath(dst_PATH), filename)
        img_scale.save(full_outfilename, format='PNG')
