from modules import DinoFeaturizer
from dataset import MVTecLocoDataset
from torch.utils.data import DataLoader
import torch
from sampler import GreedyCoresetSampler
import torch.nn.functional as F
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from crf import dense_crf
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)
i_m = i_m[:, None, None]
i_std = np.array(IMAGENET_STD)
i_std = i_std[:, None, None]

def run() -> None:
    dataset_train = MVTecLocoDataset(root_dir=dataset_root, category='train/good', resize_shape=image_size)
    dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    dataset_test_logical = MVTecLocoDataset(root_dir=dataset_root, category='test/logical_anomalies',
                                            resize_shape=image_size)
    dataloader_test_logical = DataLoader(dataset_test_logical, batch_size=1, shuffle=False, num_workers=0)

    dataset_test_good = MVTecLocoDataset(root_dir=dataset_root, category='test/good', resize_shape=image_size)
    dataloader_test_good = DataLoader(dataset_test_good, batch_size=1, shuffle=False, num_workers=0)

    dataset_test_stru = MVTecLocoDataset(root_dir=dataset_root, category='test/structural_anomalies',
                                         resize_shape=image_size)
    dataloader_test_stru = DataLoader(dataset_test_stru, batch_size=1, shuffle=False, num_workers=0)

    net = DinoFeaturizer()
    net = net.cuda()
    train_feature_list=[]
    greedsampler_perimg = GreedyCoresetSampler(percentage=0.01,device='cuda')
    if StartTrain:
        for i, Img in enumerate(dataloader):
            with torch.no_grad():
                image = Img['image']
                image = image.cuda()
                feats0, f_lowdim = net(image)
                feats = feats0.squeeze()
                feats = feats.reshape(feats0.shape[1],-1).permute(1,0)
                feats_sample = greedsampler_perimg.run(feats)
                train_feature_list.append(feats_sample)


        train_features = torch.cat(train_feature_list,dim=0)
        train_features = F.normalize(train_features, dim=1)
        torch.save(train_features.cpu(),f'{classname}.pth')
        train_features = train_features.cpu().numpy()
        kmeans=KMeans(init='k-means++',n_clusters=num_cluster)
        c = kmeans.fit(train_features)
        cluster_centers = torch.from_numpy(c.cluster_centers_)
        torch.save(cluster_centers.cpu(),f'{classname}_k{num_cluster}.pth')
        train_features_sampled = cluster_centers.cuda()
        train_features_sampled = train_features_sampled.unsqueeze(0).unsqueeze(0)
        train_features_sampled = train_features_sampled.permute(0, 3, 1, 2)
    else:
        train_features = torch.load(f'{classname}.pth').cuda()
        train_features = train_features.cpu().numpy()
        kmeans=KMeans(init='k-means++',n_clusters=num_cluster)
        c = kmeans.fit(train_features)
        cluster_centers = torch.from_numpy(c.cluster_centers_)
        train_features_sampled = cluster_centers.cuda()
        train_features_sampled = train_features_sampled.unsqueeze(0).unsqueeze(0)
        train_features_sampled = train_features_sampled.permute(0, 3, 1, 2)

    savepath = f'{classname}_heat'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    train_savepath = f'{savepath}/train'
    if not os.path.exists(train_savepath):
        os.makedirs(train_savepath)
    save_img(dataloader, train_features_sampled, net, train_savepath)

    test_savepath = f'{savepath}/test/logical_anomalies'
    if not os.path.exists(test_savepath):
        os.makedirs(test_savepath)
    save_img(dataloader_test_logical, train_features_sampled, net, test_savepath)

    test_savepath = f'{savepath}/test/good'
    if not os.path.exists(test_savepath):
        os.makedirs(test_savepath)
    save_img(dataloader_test_good, train_features_sampled, net, test_savepath)

    test_savepath = f'{savepath}/test/stru'
    if not os.path.exists(test_savepath):
        os.makedirs(test_savepath)
    save_img(dataloader_test_stru, train_features_sampled, net, test_savepath)

def save_img(dataloader, train_features_sampled, net, savapath):
    for i, Img in enumerate(dataloader):
        image = Img['image']
        imageo = Img['image1'][0,:,:,:]
        imageo = unloader(imageo)
        heatmap,heatmap_intra = get_heatmaps(image, train_features_sampled, net)
        img_savepath = f'{savapath}/{i}'
        if not os.path.exists(img_savepath):
            os.makedirs(img_savepath)
        imageo.save(f'{img_savepath}/imgo.jpg')
        see_image(image, heatmap,img_savepath,heatmap_intra)

def get_heatmaps(img,query_feature,net):
    with torch.no_grad():
        feats1, f1_lowdim = net(img.cuda())
    sfeats1 = query_feature
    attn_intra = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1))
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)
    heatmap_intra = F.interpolate(
        attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
    img_crf = img.squeeze()
    crf_result = dense_crf(img_crf,heatmap_intra)
    heatmap_intra = torch.from_numpy(crf_result)
    d = heatmap_intra.argmax(dim=0)
    d = d[None, None, :, :]
    d = d.repeat(1,3,1,1)
    seg_map = torch.zeros([1, 3, d.shape[2], d.shape[3]], dtype=torch.int64)
    for color in range(query_feature.shape[3]):
        seg_map = torch.where (d==color, color_tensor[color], seg_map )
    return seg_map,heatmap_intra

def see_image(data,heatmap,savepath,heatmap_intra):
    data = data[0, :, :, :]
    data =data.cpu().numpy()
    data = np.clip((data*i_std+i_m)*255,0,255).astype(np.uint8)
    data = data.transpose(1,2,0)
    data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{savepath}/img.jpg',data)

    heatmap = heatmap[0, :, :, :].cpu().numpy()
    heatmap = heatmap.transpose(1,2,0)
    cv2.imwrite(f'{savepath}/heatresult.jpg', heatmap)

    for i in range(heatmap_intra.shape[0]):
        heat=heatmap_intra[i,:,:].cpu().numpy()
        heat = np.round(heat*128).astype(np.uint8)
        cv2.imwrite(f'{savepath}/heatresult{i}.jpg', heat)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', action='store', type=str, required=True)
    args = parser.parse_args()

    unloader = transforms.ToPILImage()
    StartTrain = True
    image_size = 224
    color_list = [[127, 123, 229], [195, 240, 251], [146, 223, 255], [243, 241, 230], [224, 190, 144], [178, 116, 75]]
    color_tensor = torch.tensor(color_list)
    color_tensor = color_tensor[:, :, None, None]
    color_tensor = color_tensor.repeat(1, 1, image_size, image_size)
    num_cluster = 5

    classlist = ['screw_bag','breakfast_box', 'juice_bottle', 'pushpins', 'splicing_connectors', ]
    for classname in classlist:
        dataset_root = f'{args.datasetpath}/{classname}/'
        run()