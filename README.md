Implementaion of the U-Net/Swin-Unet with attention techniques including attention gate (AG), bottleneck attention module (BAM), convolutional block attention module (CBAM), squeeze-and-exitation (SE) and selective-kernel (SK). Support shortcut connection, multi-scale feature fusion, scale-adaptive feature selection (SAFS). Support cross-entropy loss, structural similarity (SSIM) loss, Dice loss and IoU loss ...

## References

AG: https://arxiv.org/abs/1808.08114

BAM: https://arxiv.org/abs/1807.06514

CBAM: https://arxiv.org/abs/1807.06521

SE: https://arxiv.org/abs/1709.01507?spm=a2c41.13233144.0.0

SK: https://arxiv.org/abs/1903.06586

Swin-Unet: https://arxiv.org/abs/2105.05537

## Datasets

PAIP 2019 https://paip2019.grand-challenge.org/Dataset/

PAIP 2020 https://paip2020.grand-challenge.org/Dataset/

FIVES 2021 https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1?file=34969398

MoNuSeg 2018 https://monuseg.grand-challenge.org/Data/

## whole-slide image segmentation

### PAIP 2019
![2019](/results/segmap.jpeg)

### PAIP 2020
![2020](/results/segmap2020.jpeg)

## retinal vessel segmentation
![fives](/results/fives_output.png)

## nuclei segmentation
![monuseg](/results/monuseg_output.png)

...

We kindly refer the reader to our articles for further explanation: 

1. Scale-adaptive viable tumor burden estimation via histopathological microscopy image segmentation

https://www.sciencedirect.com/science/article/abs/pii/S0010482525002665

2. Structure-aware scale-adaptive networks for cancer segmentation in whole-slide images

https://arxiv.org/abs/2109.12617
