# ComAD
This is the official code for the paper entitled "Component-aware anomaly detection framework for adjustable and logical industrial visual inspection"



## Erratum
The DINO features used in this paper are extracted from the last Transformer block, rather than the first block as incorrectly stated in the text. Additionally, the corresponding model size is 21M parameters, not 1.9MB.


Due to the randomness of KMeans, the results of each experiment will vary slightly. For the original paper, we ran a total of five times and took the average value. 

If you have any questions, you could also contact ltk98633@stu.xjtu.edu.cn

## Preparation
Our selected benchmarks include the following:

MVTec Loco AD dataset: https://www.mvtec.com/company/research/datasets/mvtec-loco.

CAD-AD dataset: https://github.com/IshidaKengo/SA-PatchCore

## Pipeline:
First run:

```python seg_image.py --datasetpath .../mvtec_loco/```to segment the image. Change the ```--datasetpath``` to your own file path.

Then run (make sure you have previously finished seg_image.py)

```python logical_anomaly_detection.py``` to achieve logical anomaly detection


## Citation 
If you find this work helpful to your project, please cite
```
@article{liu2023component,
  title={Component-aware anomaly detection framework for adjustable and logical industrial visual inspection},
  author={Liu, Tongkun and Li, Bing and Du, Xiao and Jiang, Bingke and Jin, Xiao and Jin, Liuyi and Zhao, Zhuo},
  journal={Advanced Engineering Informatics},
  volume={58},
  pages={102161},
  year={2023},
  publisher={Elsevier}
}
```
# Acknowledgement
We use some codes from https://github.com/mhamilton723/STEGO, https://github.com/facebookresearch/dino, https://github.com/amazon-science/patchcore-inspection, and https://github.com/VitjanZ/DRAEM. A big thanks to their great work!



