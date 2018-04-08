import numpy as np
import scipy.misc as sm


def gen_pred_img(ffg, fbg, lfg):
    """shape = gt.shape # [t, h, w, c]
                img = np.zeros([shape[1]*2, shape[2]*shape[0], shape[3]])
                for i in range(shape[0]):
                    img[:shape[1], i*shape[2]:(i+1)*shape[2]] = gt[i]
                img[shape[1]:, :shape[2]] = ffg
                img[shape[1]:, shape[2]:2*shape[2]] = fbg
                img[shape[1]:, -shape[2]:] = lfg"""
    border = 2
    shape = ffg.shape # [h, w, c]
    image = np.ones([shape[0]+2*border, shape[1]*3+4*border, shape[2]]) * 255
    image[border:-border,border:shape[1]+border] = ffg
    image[border:-border,shape[1]+border*2:2*shape[1]+border*2] = fbg
    image[border:-border,2*shape[1]+3*border:-border] = lfg

    return image

def gen_pred_vid(vid):
    shape = vid.shape
    vid_img = np.zeros((shape[1], shape[0]*shape[2], shape[3]))

    for i in range(shape[0]):
        vid_img[:,i*shape[2]:(i+1)*shape[2]] = vid[i]

    return vid_img


def add_patch_to_image(image, patch, x, y):
    ih, iw = image.shape[0], image.shape[1]
    ph, pw = patch.shape[0], patch.shape[1]

    if x >= iw or y >= ih: return image

    ch, cw = ih - y, iw - x
    if ch > ph: ch = ph
    if cw > pw: cw = pw

    patch = patch[:ch,:cw]
    image[y:y+ch,x:x+cw] = patch

    return image


def save_images(clip):
    # just a temporary function to ensure we read tf-records correctly
    for i in range(16):
        sm.imsave('%d.jpg'%i, clip[i])


def save_fg_bg(clip, mask):
    # just a temporary function to ensure we read tf-records correctly
    mask = np.stack([mask, mask, mask], -1)
    for i in range(16):
        sm.imsave('fg_%d.jpg'%i, clip[i]*mask[i])
        sm.imsave('bg_%d.jpg'%i, clip[i]*(1-mask[i]))