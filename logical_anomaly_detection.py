import glob
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from utils_area import compute_imagewise_retrieval_metrics, get_area_list_new, get_area_only_histo, train_select_binary_offsets, test_select_binary_offsets
from filter_algorithm import filter_bg_noise


def test_area_color_component(area_tegood_all, color_tegood_all, files, sub, k_offset, train_mean, train_std, cmean,
                              cstd, dbscan, nn_connection):
    area_tegood, color_tegood, _, _ = test_select_binary_offsets(files, sub, k_offset, train_mean, train_std, cmean,
                                                                 cstd)
    area_tegood_all.append(area_tegood)
    color_tegood_all.append(color_tegood)
    # test_good_component
    tegood_histo_numpy = get_area_only_histo(files, sub, k_offset, dbscan, nn_connection)
    return area_tegood_all, color_tegood_all, tegood_histo_numpy


def test_global_info(score_all, area_tegood_all, color_tegood_all, nn_train_global):
    area_tegood_all_numpy = np.concatenate(area_tegood_all, axis=1)
    color_tegood_all_numpy = np.concatenate(color_tegood_all, axis=1)
    tegood_global = np.concatenate((area_tegood_all_numpy, color_tegood_all_numpy), axis=1)
    dis_tegood_global, _ = nn_train_global.kneighbors(tegood_global)
    dis_tegood_global = np.mean(dis_tegood_global, axis=1)
    score_tegood_all = score_all + dis_tegood_global
    return score_tegood_all

subdict={}
class_num = 0
auroc_log = 0
auroc_stru = 0
auroc_all = 0
classlist = ['breakfast_box', 'juice_bottle', 'screw_bag', 'pushpins', 'splicing_connectors']
sourcepath = '.'
for classname in classlist:
    train_file_path = f'{sourcepath}/{classname}_heat/train'
    trainfiles = sorted(glob.glob(train_file_path+'/*'), key=lambda x: int(x.split('/')[-1]))

    tegood_file_path = f'{sourcepath}/{classname}_heat/test/good'
    tegoodfiles = sorted(glob.glob(tegood_file_path+'/*'), key=lambda x: int(x.split('/')[-1]))

    telogical_file_path = f'{sourcepath}/{classname}_heat/test/logical_anomalies'
    telogicalfiles = sorted(glob.glob(telogical_file_path+'/*'), key=lambda x: int(x.split('/')[-1]))

    testrufile_path = f'{sourcepath}/{classname}_heat/test/stru'
    testrufiles = sorted(glob.glob(testrufile_path+'/*'), key=lambda x: int(x.split('/')[-1]))

    valfile_path = f'{sourcepath}/{classname}_heat/test/validation'
    valfiles = sorted(glob.glob(valfile_path+'/*'), key=lambda x: int(x.split('/')[-1]))
    knn = 5
    score_tegood_all = 0
    score_telogical_all = 0
    score_testru_all = 0
    score_val_all = 0
    area_train_all = []
    area_tegood_all = []
    area_tlogical_all =[]
    area_stru_all = []
    area_val_all = []
    color_train_all = []
    color_tegood_all =[]
    color_tlogical_all = []
    color_stru_all = []
    color_val_all = []

    #component
    tgoodhis_all = []
    tloghis_all = []
    trainhis_all = []
    subdict[f'{classname}'] = filter_bg_noise(sourcepath,classname)
    for sub in subdict[f'{classname}']:
        #train_area
        area_train, train_mean, train_std, k_offset = train_select_binary_offsets(trainfiles, sub)
        nn_area = NearestNeighbors(n_neighbors=knn)
        nn_area.fit(area_train)
        area_train_all.append(area_train)

        #train component
        component_area_list = get_area_list_new(trainfiles, sub, k_offset)
        component_area = np.asarray(component_area_list)
        component_area_mean = np.mean(component_area)
        dbscan_r = component_area_mean*0.1
        dbscan_min = 10
        dbscan = DBSCAN(eps=dbscan_r, min_samples=dbscan_min)

        dbscan.fit(component_area)
        nn_connection = NearestNeighbors(n_neighbors=knn)
        nn_connection.fit(component_area)
        train_histo_numpy = get_area_only_histo(trainfiles, sub, k_offset, dbscan, nn_connection)
        nn_connection_histo = NearestNeighbors(n_neighbors=knn)
        nn_connection_histo.fit(train_histo_numpy)
        #train_color
        _, color_train, cmean, cstd = test_select_binary_offsets(trainfiles, sub, k_offset, train_mean, train_std, 0 ,0)
        color_train_all.append(color_train)

        #test_good
        area_tegood_all, color_tegood_all, tegood_histo_numpy = test_area_color_component(area_tegood_all,color_tegood_all,tegoodfiles, sub, k_offset, train_mean, train_std, cmean, cstd,dbscan, nn_connection)
        #test_logical
        area_tlogical_all, color_tlogical_all, telogical_histo_numpy = test_area_color_component(area_tlogical_all,color_tlogical_all,telogicalfiles, sub, k_offset, train_mean, train_std, cmean, cstd,dbscan, nn_connection)
        #test_stru
        area_stru_all, color_stru_all, tstru_histo_numpy = test_area_color_component(area_stru_all, color_stru_all,testrufiles, sub, k_offset, train_mean, train_std, cmean, cstd,dbscan, nn_connection)

        trainhis_all.append(train_histo_numpy)
        # compont test good error
        alpha = 0.5
        dis_tegood, indices = nn_connection_histo.kneighbors(tegood_histo_numpy)
        dis_tegood = np.mean(dis_tegood, axis=1)*alpha/(train_histo_numpy.shape[1]*train_histo_numpy.shape[1])
        score_tegood_all = score_tegood_all + dis_tegood
        tgoodhis_all.append(tegood_histo_numpy)

        # componet test logical error
        dis_telogical, indices = nn_connection_histo.kneighbors(telogical_histo_numpy)
        dis_telogical = np.mean(dis_telogical, axis=1)*alpha/(train_histo_numpy.shape[1]*train_histo_numpy.shape[1])
        score_telogical_all = score_telogical_all + dis_telogical
        tloghis_all.append(telogical_histo_numpy)

        #component test stru error
        dis_testru, indices = nn_connection_histo.kneighbors(tstru_histo_numpy)
        dis_testru = np.mean(dis_testru, axis=1)*alpha/(train_histo_numpy.shape[1]*train_histo_numpy.shape[1])
        score_testru_all = score_testru_all + dis_testru

    area_train_all_numpy = np.concatenate(area_train_all, axis=1)
    color_train_all_numpy = np.concatenate(color_train_all, axis=1)
    train_global = np.concatenate((area_train_all_numpy, color_train_all_numpy), axis=1)
    nn_train_global = NearestNeighbors(n_neighbors=knn)
    nn_train_global.fit(train_global)

    #testgood
    score_tegood_all = test_global_info(score_tegood_all, area_tegood_all, color_tegood_all, nn_train_global)

    #testlogical
    score_telogical_all = test_global_info(score_telogical_all, area_tlogical_all, color_tlogical_all, nn_train_global)


    #teststru
    score_testru_all = test_global_info(score_testru_all, area_stru_all, color_stru_all, nn_train_global)

    gt_test_good = [False for i in range(score_tegood_all.shape[0])]
    gt_test_logical = [True for i in range(score_telogical_all.shape[0])]
    gt_test_stru = [True for i in range(score_testru_all.shape[0])]

    all_socres = np.concatenate((score_tegood_all, score_telogical_all, score_testru_all), axis=0).astype(np.float32)
    all_socres_onlylo = np.concatenate((score_tegood_all, score_telogical_all), axis=0).astype(np.float32)
    all_socres_onlystru = np.concatenate((score_tegood_all, score_testru_all), axis=0).astype(np.float32)

    all_labels_onlog = gt_test_good + gt_test_logical
    all_labels_onlystu = gt_test_good + gt_test_stru
    all_labels = gt_test_good + gt_test_logical + gt_test_stru

    auroc_alllog = compute_imagewise_retrieval_metrics(all_socres_onlylo, all_labels_onlog)['auroc']
    auroc_onlystu = compute_imagewise_retrieval_metrics(all_socres_onlystru, all_labels_onlystu)['auroc']
    auroc_allscore = compute_imagewise_retrieval_metrics(all_socres, all_labels)['auroc']
    print(f'{classname} auroc_logical: {auroc_alllog} auroc_stru: {auroc_onlystu} auroc_all: {auroc_allscore}')
    class_num = class_num + 1
    auroc_log = auroc_log + auroc_alllog
    auroc_stru = auroc_stru + auroc_onlystu
    auroc_all = auroc_all + auroc_allscore

auroc_log = auroc_log /class_num
auroc_stru = auroc_stru /class_num
auroc_all = auroc_all /class_num
print(f'average logical {auroc_log}, average stru {auroc_stru}, average all {auroc_all}')
