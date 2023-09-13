# ComAD
This is the official code for the paper entitled "Component-aware anomaly detection framework for adjustable and logical industrial visual inspection"
The latest version is on https://www.sciencedirect.com/science/article/abs/pii/S1474034623002896
The old version: https://arxiv.org/abs/2305.08509

Due to the randomness of KMeans, the results of each experiment will vary. For the original paper, we ran a total of five times and took the average value. 

If you have any questions, you could also contact ltk98633@stu.xjtu.edu.cn

# Preparation
You need to first download MVTec Loco AD dataset https://www.mvtec.com/company/research/datasets/mvtec-loco.

# Run seg_image.py
python seg_image.py --datasetpath .../mvtec_loco/

# Run logical_anomaly_detection.py (make sure you have previously finished seg_image.py)
python logical_anomaly_detection.py
# Acknowledgement
We use some codes from https://github.com/mhamilton723/STEGO, https://github.com/facebookresearch/dino, https://github.com/amazon-science/patchcore-inspection, and https://github.com/VitjanZ/DRAEM. A big thanks to their great work!


