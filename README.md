# Learning-based Homography Matrix Optimization for Dual-fisheye Video Stitching
Mufeng Zhu, Yang Sui, Bo Yuan, Yao Liu

This repository contains the official authors implementation associated with the paper ["Learning-based Homography Matrix Optimization for Dual-fisheye Video Stitching"](https://dl.acm.org/doi/abs/10.1145/3609395.3610600). We provide source code for stitching dual-fisheye videos.

Abstract: In this paper, we propose a novel feature-based video stitching algorithm for stitching back-to-back fisheye camera videos into one omnidirectional video in a video live streaming scenario. Our main contribution lies in a learning-based approach that refines the homography matrix in an online manner via gradient descent. The homography matrix is updated by training on a rolling dataset of feature points that are extracted and matched as new video frames are captured. Experimental results show that our method can create stitched images that better align matching features with lower mean squared error (MSE) than traditional feature-based stitching method. Furthermore, compared to vendor-supplied software (VUZE VR Studio) that uses calibration-based stitching, our method also produces visibly better results.
## BibTex
```bash
@inproceedings{10.1145/3609395.3610600,
author = {Zhu, Mufeng and Sui, Yang and Yuan, Bo and Liu, Yao},
title = {Learning-based Homography Matrix Optimization for Dual-fisheye Video Stitching},
year = {2023},
isbn = {9798400703034},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3609395.3610600},
doi = {10.1145/3609395.3610600},
booktitle = {Proceedings of the 2023 Workshop on Emerging Multimedia Systems},
pages = {48–53},
numpages = {6},
keywords = {fisheye, omnidirectional video stitching, feature extraction, homography matrix optimization},
location = {New York, NY, USA},
series = {EMS '23}
}
```
## Dependencies
OpenCV-4.6

Numpy

Pytorch

## Running in Linux
```bash
python Fisheye_stitching.py -o output_file_path -i1 left_fisheye_video -i2 right_fisheye_video -f FOV
```

We also provide test videos. You can download [here](https://drive.google.com/drive/u/0/folders/1GeUMMLfAjMSx9PY3jvAlSngmlrBlSwko)

## References
Our source code comes from [this repository](https://github.com/cynricfu/dual-fisheye-video-stitching). Thanks for the help.

[Dual-fisheye lens stitching for 360-degree imaging](https://arxiv.org/pdf/1708.08988.pdf)
