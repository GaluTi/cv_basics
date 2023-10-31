from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

# sem_1

# calculate new size for image to resize it in square (max_size x max_size)
def calc_size(img_size, max_size):
    if img_size[0] > img_size[1]:
        return((max_size, int(max_size * img_size[1] / img_size[0])))
    else:
        return((int(max_size * img_size[0] / img_size[1]), max_size))

# returns a black and white version of the original picture
def make_grey(img):
    new_img = []
    for x in np.array(img):
        new_x = []
        for y in x:
            color = np.mean(y)
            new_x.append(color)
        new_img.append(new_x)
    return new_img

# draw images and their histograms
def draw_plots(images, fsize, bins=30, dens=False):
    fig, ax = plt.subplots(2, len(images), figsize=fsize)
    for i in range(len(images)):
        ax[0][i].imshow(np.array(images[i]), cmap="gray")
        ax[1][i].hist(np.array(images[i]).reshape(-1), edgecolor="black", alpha=0.5, bins=bins, density=dens)
        ax[1][i].set_xlim(0, 255)
        
# sem_2

def change_intence(img, k, b):
    img = np.array(img)
    img_ci = []
    for x in img:
        x_ci = (k*x + b).astype("uint8")
        x_ci = np.where(x_ci < 0, 0, x_ci)
        x_ci = np.where(x_ci > 255, 255, x_ci)
        img_ci.append(x_ci)
    return img_ci

def auto_contrast(img, borders=True):
    img_arr = np.array(img).reshape(-1)
    if borders:
        hist_min = np.min(img_arr)
        img_tmp = change_intence(img, 1, -hist_min)
        img_arr = np.array(img_tmp).reshape(-1)
        hist_max = np.max(img_arr)
        new_k = 255/hist_max
        return change_intence(img_tmp, new_k, 0)
    else:
        img_tmp = []
        hist_min = np.min([x for x in img_arr if x > 0])
        for x in np.array(img):
            img_tmp.append(np.where(pd.Series(x).between(1, 254), x - hist_min + 1, x))   
        
        img_arr = np.array(img_tmp).reshape(-1)
        hist_max = np.max([x for x in img_arr if x < 255])
        new_k = 253/hist_max
        new_img_tmp = []
        for x in img_tmp:
            new_img_tmp.append(np.where(pd.Series(x).between(2, 254), new_k * x, x))
        return new_img_tmp
    
def percent_correction(img, p_l, p_r):
    img_arr = np.array(img)
    pixel_num = img_arr.shape[0] * img_arr.shape[1]
    ind = np.unravel_index(np.argsort(img_arr, axis=None), img_arr.shape)
    n_l = int(pixel_num * p_l)
    n_r = int(pixel_num * p_r)
    print(n_l, n_r)
    
    tmp_img = copy.copy(img_arr)
    for i in range(n_l):
        tmp_img[ind[0][i]][ind[1][i]] = 255
    for i in range(pixel_num - n_r, pixel_num):
        tmp_img[ind[0][i]][ind[1][i]] = 0
    
    return auto_contrast(tmp_img, False)

# sem_3

def calc_g_lut(g):
    lut = []
    for i in range(256):
        lut.append( ((i/255)**g) * 255 )
    return lut

def g_cor(img, g):
    lut = calc_g_lut(g)
    new_img = []
    for x in img:
        new_x = []
        for y in x:
            new_x.append(lut[int(y)])
        new_img.append(new_x)
    return new_img

# sem_4

def sigma_b(img, t, bins):
    hist = np.histogram(np.array(img).reshape(-1), bins=bins)
    n1 = sum([hist[0][i] for i in range(t)])
    n2 = sum([hist[0][i] for i in range(t, len(hist[0]))])
    mu1 = sum([hist[0][i]*hist[1][i] for i in range(t)]) / n1
    mu2 = sum([hist[0][i]*hist[1][i] for i in range(t, len(hist[0]))]) / n2
    return n1 *n2 * (mu1 - mu2)**2

def otsu_alg(img, bins):
    t_optim = 0
    s_b = -1
    hist = np.histogram(np.array(img).reshape(-1), bins=bins)
    for t in range(1, len(hist[0])):
        s_b_tmp = sigma_b(img, t, bins=bins)
        if s_b_tmp > s_b:
            s_b = s_b_tmp
            t_optim = t
    return hist[1][t_optim]

def bin_img(img, t):
    img_arr = np.array(img)
    return np.where(img_arr <= t, 0, 1)

# sem_5

def cummulative_summ(matrix):
    c_matrix = copy.copy(matrix)
    for i in range(1, len(matrix[0])):
        c_matrix[0][i] += c_matrix[0][i-1]
        c_matrix[i][0] += c_matrix[i-1][0]
    for i in range(1, len(matrix[0])):
        for j in range(1, len(matrix[0])):
            c_matrix[i][j] = c_matrix[i][j] + c_matrix[i - 1][j] + c_matrix[i][j - 1] - c_matrix[i - 1][j - 1]
    return c_matrix

def make_window(arr, i, j, d):
    wnd = []
    for k in np.arange(-d, d + 1):
        x = []
        for l in np.arange(-d, d + 1):
            x.append(arr[i + k][j + l])
        wnd.append(x)
    return wnd

def image_windows(img, a):
    d = int((a - 1) / 2)
    windows = []
    img_arr = np.array(img)
    img_arr_pad = np.pad(img_arr, (d, d), "edge")
    for i in range(d, len(img_arr_pad) - d):
        wnd_list = []
        for j in range(d, len(img_arr_pad[0]) - d):
            wnd = make_window(img_arr_pad, i, j, d)
            wnd_list.append(wnd)
        windows.append(wnd_list)
    return windows

def image_means(windows, a):
    means = []
    for wnd_list in windows:
        x = []
        for wnd in wnd_list:
            cs = cummulative_summ(wnd)
            x.append(cs[a - 1][a - 1] / a**2)
        means.append(x)
    return means

def image_sigmas(windows, img_means, a):
    sigmas = []
    img_means2 = image_means(np.square(windows), a)
    for (x1, x2)  in zip(img_means, img_means2):
        x = []
        for (y1, y2)  in zip(x1, x2):
            x.append(y2 - y1**2)
        sigmas.append(x)
    return sigmas

def cult_t_lb(img_means, img_sigmas, k):
    T = []
    for (m_list, s_list) in zip(img_means, img_sigmas):
        t = []
        for (m, s) in zip(m_list, s_list):
            t.append(m + k*np.sqrt(s))
        T.append(t)
    return T

def local_bin(img, t_lb):
    img_arr = np.array(img)
    img_lb = []
    for (x_list, t_list) in zip(img_arr, t_lb):
        img_lb.append(np.where(x_list <= t_list, 0, 255))
    return img_lb