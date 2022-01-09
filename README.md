# R2IPoints : Pursuing Rotation-Insensitive Point Representation for Aerial Object Detection
Codes for R2IPoints : Pursuing Rotation-Insensitive Point Representation for Aerial Object Detection.

### The overview of R2IPoints:
![overview](https://github.com/shnew/R2IPoints/blob/main/img/overview.png)
# Getting Started  
### Prerequisties
* pip install torch==1.1.0  
* pip install torchvision==0.3.0  
* pip install mmdet==1.2.0  
* pip install mmcv==0.3.1  
* pip install -r requirements/build.txt
* pip install -v -e . 
### Datasets preparation
* DIOR: http://www.escience.cn/people/gongcheng/DIOR.html
* DOTA: https://captain-whu.github.io/DOTA/
  * code for cutting images: https://github.com/dingjiansw101/AerialDetection/tree/master/DOTA_devkit
### Quick start
* python tools/train.py ./configs/R2IPoints/R2IPoints_moment_r101_fpn_2x_DIOR.py 
* python tools/test.py  ./configs/R2IPoints/R2IPoints_moment_r101_fpn_2x_DIOR.py ${CHECKPOINT_FILE} --eval mAP
