import cv2
import numpy as np
import glob

def filter_bg_noise(sourcepath, classname):
    train_file_path = f'{sourcepath}/{classname}_heat/train'
    trainfiles = sorted(glob.glob(train_file_path+'/*'), key=lambda x: int(x.split('/')[-1]))
    img0_path = trainfiles[0]
    reserve_list = []
    max_list = []
    seg_img_list = sorted(glob.glob(img0_path+'/*[heatresult][0-9].jpg'))
    for i,imgpath in enumerate(seg_img_list):
        gray_img = cv2.imread(imgpath, 0)
        gray_cal_otsu = gray_img[10:gray_img.shape[0]-10, 10:gray_img.shape[0]-10]
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, thresh2 = cv2.threshold(gray_img, ret, 1, cv2.THRESH_BINARY)
        corner1 = np.zeros_like(thresh2)
        corner1[0:20, 0:20] = 1

        corner2 = np.zeros_like(thresh2)
        corner2[204:224, 0:20] = 1

        corner3 = np.zeros_like(thresh2)
        corner3[0:20, 204:224] = 1

        corner4 = np.zeros_like(thresh2)
        corner4[204:224, 204:224] = 1

        ex = (corner1*thresh2).max() + (corner2*thresh2).max() + (corner3*thresh2).max() + (corner4*thresh2).max()
        kenel_size = (11,11)
        gray_img = cv2.blur(gray_img, kenel_size)
        maxvalue = gray_img.max()
        max_list.append(maxvalue)
        if maxvalue >64 and ex<3: #when save the heatmap, the normalized heatmap was multipled by 128, here the threshold is set as 0.5*128 = 64
            reserve_list.append(i)
    return reserve_list
