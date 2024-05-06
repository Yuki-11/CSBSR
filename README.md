# 
<p align="center">
  <h1 align="center"> [IEEE TIM'24] <ins>CSBSR</ins>:<br> Joint Learning of Blind Super-Resolution and
Crack Segmentation for Realistic Degraded Images</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=lijAs5AAAAAJ">Yuki Kondo</a>
    ·
    <a href="https://scholar.google.com/citations?us
    er=Tgbsbs8AAAAJ">Norimichi Ukita</a>
  </p>
  <p align="center">
  Toyota Technological Institute<br>
  IEEE Transactions on Instrumentation and Measurement (TIM) 2024
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2302.12491" align="center">Paper</a> | 
    <a href="https://yuki-11.github.io/CSBSR-project-page/" align="center">Project Page</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://raw.githubusercontent.com/Yuki-11/CSBSR/main/fig/result_csbsr_github.png" alt="example" width=100%>
    <br>
    <em>Crack segmentation challenges for synthetically-degraded images given by low resolution and anisotropic Gaussian blur. Our method (f) CSBSR succeeds in detecting cracks in the most detail compared to previous studies (d), (e). Furthermore, in several cases our method was able to detect cracks as successfully as when GT high-resolution images were used for segmentation (c), despite the fact that our method was inferring from degraded images.</em>
</p>

-----

<p align="center">
    <img src="https://raw.githubusercontent.com/Yuki-11/CSBSR/main/fig/csbsr-arch.png" alt="example" width=100%>
    <br>
    <em>Proposed joint learning network with blind SR and segmentation.</em>
</p>

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/Yuki-11/CSBSR/blob/main/LICENSE)


## Table of Contents

- [About the Project](#about-the-project)
- [Usage](#usage)
- [License](#license)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)
- [Contact](#contact)

## About the Project
This paper proposes crack segmentation augmented by super resolution (SR) with deep neural networks. In the proposed method, a SR network is jointly trained with a binary segmentation network in an end-to-end manner. This joint learning allows the SR network to be optimized for improving segmentation results. For realistic scenarios, the SR network is extended from non-blind to blind for processing a low-resolution image degraded by unknown blurs. The joint network is improved by our proposed two extra paths that further encourage the mutual optimization between SR and segmentation. Comparative experiments with State-of-the-Art segmentation methods demonstrate the superiority of our joint learning, and various ablation studies prove the effects of our contributions.

## Usege

### Setup Environment

```bash
git clone git@github.com:Yuki-11/CSBSR.git
cd CSBSR
virtualenv env
source env/bin/activate
pip install -r requirement.txt
```

Due to [a temporary issue with wandb](https://community.wandb.ai/t/failed-to-import-wandb-after-updating-to-wandb-0-16-6/6361/6), specify the version of protobuf:

```bash
codepip install protobuf==3.20.1
```

### Download Model Weights

Download the model weights from [here](https://drive.google.com/drive/folders/1dbLbf4vl_O4OFfzJ3vFwFQ187fAZeI5X) and save them under /weights/. The performance of the released models is as follows.


|Model|IoU<sub>max</sub>|AIU|HD95<sub>min</sub>|AHD95|PSNR|SSIM|Link|
|:----|:----|:----|:----|:----|:----|:----|:----|
|CSBSR w/ PSPNet (*β*=0.3)|0.573|0.552|20.92|22.52|28.75|0.703|[here](https://drive.google.com/drive/folders/116EvkbVlOfwB8AfrLmtsVVJOxtMlCFpw?usp=drive_link)
|CSBSR w/ HRNet+OCR (*β*=0.9)|0.553|0.534|17.54|20.29|27.66|0.668|[here](https://drive.google.com/drive/folders/1wsnGw6440eg-Y7QCtTQntxrWl7wILmTr?usp=drive_link)
|CSBSR w/ CrackFormer (*β*=0.9)|0.469|0.443|39.37|56.59|25.93|0.571|[here](https://drive.google.com/drive/folders/1PpEtTfFc55ZqNXHnBVEnQWtehj003Yys?usp=drive_link)
|CSBSR w/ U-Net (*β*=0.3)|0.530|0.506|26.33|27.24|28.68|0.702|[here](https://drive.google.com/drive/folders/16v3x3o-l08UHzdQ47kG6UIu_fA91r9Hi?usp=drive_link)
|CSSR w/ PSPNet (*β*=0.7)|0.557|0.539|21.20|24.74|28.35|0.656|[here](https://drive.google.com/drive/folders/116EvkbVlOfwB8AfrLmtsVVJOxtMlCFpw?usp=drive_link)
|CSBSR w/ *w<sup>F</sup>* (*m<sup>F</sup>*=1)|0.573|0.551|18.73|21.7|28.73|0.702|[here](https://drive.google.com/drive/folders/1UJcWFGNEXG4RElrWih9uzFNY-jsjk865?usp=drive_link)
|CSBSR w/ *w<sup>F</sup>* (*m<sup>F</sup>*=1)+BlurSkip|0.550|0.528|18.06|19.1|28.65|0.702|[here](https://drive.google.com/drive/folders/1Gk_rY9YtObMh5jK5R6KB5e9ENsofvPKM?usp=drive_link)

### Prepare Dataset
Download the crack segmentation dataset by khanhha from [here](https://drive.google.com/open?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP) and extract it under datasets/. For more details, refer [khanhha's repository](https://github.com/khanhha/crack_segmentation).

Additionally, download the degraded original test set from [here](https://drive.google.com/drive/folders/1wywKIh9Z-XYwRlw1f2mNnjm8OdwnGPWg?usp=drive_link) and extract it under datasets/crack_segmentation_dataset. Finally, the directory structure should look like this:

```bash
├── datasets
│   └── crack_segmentation_dataset
│       ├── images
│       ├── masks
│       ├── readme
│       ├── test
│       ├── test_blurred
│       └── train
```

### Testing
Run the following command for testing:
```bash
python test.py weights/CSBSR_w_PSPNet_beta03/latest --sf_save_image
```

### Training
Run the following command for training:
```bash
python train.py --config_file config/config_csbsr_pspnet.yaml
```

To resume training from a checkpoint, use the following command:
```bash
python train.py --config_file output/CSBSR/model_compe/CSBSR_w_PSPNet_beta03/config.yaml --resume_iter <resume iteration>
```

## License

Distributed under the Apache-2.0 license License. Some of the code is based on the reference codes listed in the "Acknowledgements" section, and some of these reference codes have MIT license or Apache-2.0 license. Please, see `LICENSE` for more information.


## Citation
If you find our models useful, please consider citing our paper!
```
@article{kondo2024csbsr,
  author={Kondo, Yuki and Ukita, Norimichi},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images}, 
  year={2024},
  volume={73},
  number={},
  pages={1-16},
}
```

## Acknowledgement
This implementation utilizes the following code. We deeply appreciate the authors for their open-source codes.

* [Lextal/pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch)
* [Dootmaan/DSRL](https://github.com/Dootmaan/DSRL)
* [openseg-group/openseg.pytorch](https://github.com/openseg-group/openseg.pytorch)
* [LouisNUST/CrackFormer-II](https://github.com/LouisNUST/CrackFormer-II)
* [lterzero/DBPN-Pytorch](https://github.com/alterzero/DBPN-Pytorch)
* [Yuki-11/KBPN](https://github.com/Yuki-11/KBPN)
* [vinceecws/SegNet_PyTorch](https://github.com/vinceecws/SegNet_PyTorch)
* [khanhha/crack_segmentation](https://github.com/khanhha/crack_segmentation)
* [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
* [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)
* [google-deepmind/surface-distance](https://github.com/google-deepmind/surface-distance)
* [hubutui/DiceLoss-PyTorch](https://github.com/hubutui/DiceLoss-PyTorch)
* [JunMa11/SegLossOdyssey](https://github.com/JunMa11/SegLossOdyssey/tree/master)
* [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master)
* [xingyizhou/CenterNet](https://github.com/amdegroot/ssd.pytorch/)
* [YutaroOgawa/pytorch_advanced](https://github.com/YutaroOgawa/pytorch_advanced/)

## Contact

If you have any questions, feedback, or suggestions regarding this project, feel free to reach out:

- **Email:** [yuki.kondo.ab@gmail.com](mailto:yuki.kondo.ab@gmail.com)
