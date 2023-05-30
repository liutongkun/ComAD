import glob
import cv2
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import time
def predict_label(new_data,nn,dbscan):
    if dbscan.labels_.max() == -1:
        label_final = 0
    else:
        dis, indices = nn.kneighbors(new_data)
        dis_mean = np.mean(dis)
        c = dbscan.labels_[indices][0,:]
        d = c+1 #avoid negative -1
        counts = np.bincount(d)
        max_idx = np.argmax(counts)
        label_final = max_idx-1
    return label_final, dis_mean

def predict_label_histo(new_data,nn):
    dis, indices = nn.kneighbors(new_data)
    d = 1
    return dis

def get_area_only_histo(testlogicalfiles,subimage,k_offset,dbscan,nn):
    histo_list = []
    noise_size = 50 #224*224*0.001
    for imgfilepath in testlogicalfiles:
        img_path = f'{imgfilepath}/heatresult{subimage}.jpg'
        gray_img = cv2.imread(img_path, 0)
        gray_cal_otsu = gray_img[10:gray_img.shape[0]-10, 10:gray_img.shape[0]-10]
        ret, thresh = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, thresh2 = cv2.threshold(gray_img, int(k_offset * ret), 1, cv2.THRESH_BINARY)
        output = cv2.connectedComponentsWithStats(thresh2, 8, cv2.CV_32S)
        stats = output[2]
        if dbscan.labels_.max() != -1:
            histo = np.zeros(dbscan.labels_.max() + 1)
        else:
            histo = np.zeros(dbscan.labels_.max() + 2)
        for i in range(1, output[0]):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > noise_size:
                info = np.asarray([area])
                info = info[None, :]
                label, dis = predict_label(info, nn, dbscan)
                if label != -1:
                    histo[label] = histo[label] + 1
        histo = histo[None, :]
        histo_list.append(histo)
    histo_numpy = np.concatenate(histo_list, axis=0)
    return histo_numpy

def get_area_list_new(trainfiles, subimage, k_offset):
    arealist = []
    for imgfilepath in trainfiles:
        img_path = f'{imgfilepath}/heatresult{subimage}.jpg'
        gray_img = cv2.imread(img_path, 0)
        gray_cal_otsu = gray_img[10:gray_img.shape[0]-10, 10:gray_img.shape[0]-10]
        ret, thresh = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, thresh2 = cv2.threshold(gray_img, int(k_offset * ret), 1, cv2.THRESH_BINARY)
        output = cv2.connectedComponentsWithStats(thresh2, 8, cv2.CV_32S)
        stats = output[2]
        for i in range(1, output[0]):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50: #224*224*0.001
                arealist.append([area])
    return arealist

def train_select_binary_offsets(trainfiles, subimage):
    arealist = []
    k_offsets = [ 1.1, 1.2, 1.3, 1.4]
    for imgfilepath in trainfiles:
        img_path = f'{imgfilepath}/heatresult{subimage}.jpg'
        gray_img = cv2.imread(img_path, 0)
        gray_cal_otsu = gray_img[10:gray_img.shape[0]-10, 10:gray_img.shape[0]-10]
        ret0, thresh0 = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(gray_img,ret0 ,1,cv2.THRESH_BINARY)
        area = np.sum(thresh)
        arealist_perk = []
        arealist_perk.append(area)
        for k in k_offsets:
            ret2, thresh2 = cv2.threshold(gray_img, int(k * ret), 1, cv2.THRESH_BINARY)
            area_k = np.sum(thresh2)
            arealist_perk.append(area_k)
        arealist.append(arealist_perk)
    area_numpy = np.array(arealist)
    area_mean = np.mean(area_numpy, axis=0)
    area_std = np.std(area_numpy, axis=0)
    diversity = area_std / area_mean
    index = np.argmin(diversity)
    k_offsets_final = [1, 1.1, 1.2, 1.3, 1.4]
    area_numpy_return = area_numpy[:, index].reshape(-1,1)
    area_numpy_return = (area_numpy_return) / (area_mean[index])
    return area_numpy_return, area_mean[index], area_std[index], k_offsets_final[index]

def train_area_connect(trainfiles, subimage):
    arealist_perk = []
    for imgfilepath in trainfiles:
        img_path = f'{imgfilepath}/heatresult{subimage}.jpg'
        gray_img = cv2.imread(img_path, 0)
        gray_cal_otsu = gray_img[10:gray_img.shape[0]-10, 10:gray_img.shape[0]-10]
        ret, thresh = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        area = np.sum(thresh)
        arealist_perk.append(area)
    area_numpy = np.array(arealist_perk)
    area_mean = np.mean(area_numpy, axis=0)
    area_std = np.std(area_numpy, axis=0)
    area_numpy_return = area_numpy.reshape(-1,1)
    area_numpy_return = (area_numpy_return) / (area_mean)
    return area_numpy_return, area_mean, area_std

def test_select_binary_offsets(testfiles, subimage, k_offset, mean, std, cmean=0, cstd=0 ):
    arealist = []
    colorlist = []
    for imgfilepath in testfiles:
        img_path = f'{imgfilepath}/heatresult{subimage}.jpg'
        ori_img_path = f'{imgfilepath}/img.jpg'
        gray_img = cv2.imread(img_path, 0)
        rgb_img = cv2.imread(ori_img_path)
        imglabo = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
        gray_cal_otsu = gray_img[10:gray_img.shape[0]-10, 10:gray_img.shape[0]-10]
        ret, thresh = cv2.threshold(gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, thresh2 = cv2.threshold(gray_img, int(k_offset * ret), 1, cv2.THRESH_BINARY)
        color_sum_a = imglabo[:,:,1].astype(np.float)
        color_sum_b = imglabo[:,:,2].astype(np.float)
        color_div = (color_sum_b/(color_sum_a+0.0000001))*(color_sum_b/(color_sum_a+0.0000001))
        color_div = color_div*thresh2
        area = np.sum(thresh2)
        color_value = np.sum(color_div)/(area+0.0000001)
        arealist.append(area)
        colorlist.append(color_value)
    area_numpy_return = np.array(arealist).reshape(-1,1)
    area_numpy_return = (area_numpy_return) / (mean)
    color_numpy = np.array(colorlist).reshape(-1,1)
    if cmean != 0:
        c_mean = cmean
        c_std = cstd
    else:
        c_mean = np.mean(color_numpy, axis=0)
        c_std = np.std(color_numpy, axis=0)
    scale = 1
    color_numpy_return = (color_numpy)/(c_mean+0.0000001)*scale
    return area_numpy_return, color_numpy_return, c_mean, c_std

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}
