# SFRDP-Net
"Spatial-Frequency Residual-guided Dynamic Perceptual Network for remote sensing haze removal"
This work is available at [TGRS2025](https://ieeexplore.ieee.org/document/10892218). Official Pytorch based implementation.

## Abstract
Recently, deep neural networks have been extensively explored in remote sensing image haze removal and achieved remarkable performance. However, most existing haze removal methods fail to effectively leverage the fusion of spatial and frequency information, which is crucial for learning more representative features. Moreover, the prevalent perceptual loss used in dehazing model training overlooks the diversity among perceptual channels, leading to performance degradation. To address these issues, we propose a Spatial-Frequency Residual-guided Dynamic Perceptual Network (SFRDP-Net) for remote sensing image haze removal. Specifically, we first propose a Residual-guided Spatial-Frequency Interaction (RSFI) module, which incorporates a Bidirectional Residual Complementary Mechanism (BRCM) and a Frequency Residual Enhanced Attention (FREA). Both BRCM and FREA exploit spatial-frequency complementarity to guide more effective fusion of spatial and frequency information, thus enhancing feature representation capability and improving haze removal performance. Furthermore, a Dynamic Channel Weighting Perceptual Loss (DCWP-Loss) is developed to impose constraints with varying strengths on different perceptual channels, advancing the reconstruction of high-quality haze-free images. Experiments on challenging benchmark datasets demonstrate our SFRDP-Net outperforms several state-of-the-art haze removal methods.

## Overall architecture
![image]()

## RSFI module
![image]()


## Contents

- [Dependencies](#dependences)
- [Filetree](#filetree)
- [Pretrained Model](#pretrained-weights-and-dataset)
- [Train](#train)
- [Test](#test)
- [Clone the repo](#clone-the-repo)
- [Qualitative Results](#qualitative-results)
  - [Results on HRSD-DHID remote sensing Dehazing Challenge testing images:](#results-on-hrsd-dhid-remote-sensing-dehazing-challenge-testing-images)
  - [Results on HRSD-LHID remote sensing Dehazing Challenge testing images:](#results-on-hrsd-lhid-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thin-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-moderate-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thick-remote-sensing-dehazing-challenge-testing-images)
  - [Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:](#results-on-ntire-2021-nonhomogeneous-dehazing-challenge-testing-images)
  - [Results on RESIDE-Outdoor Dehazing Challenge testing images:](#results-on-reside-outdoor-dehazing-challenge-testing-images)
- [Copyright](#copyright)
- [Thanks](#thanks)

### Dependences

1. Pytorch 1.8.0
2. Python 3.7.1
3. CUDA 11.7
4. Ubuntu 18.04

### Filetree

```
├── README.md
├── /PSMB-Net/
|  ├── train.py
|  ├── test.py
|  ├── Model.py
|  ├── Model_util.py
|  ├── perceptual.py
|  ├── train_dataset.py
|  ├── test_dataset.py
|  ├── utils_test.py
|  ├── make.py
│  ├── /pytorch_msssim/
│  │  ├── __init__.py
│  ├── /datasets_train/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /datasets_test/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /output_result/
├── LICENSE.txt
└── /images/
```

### Pretrained Weights and Dataset

Download our model weights on Baidu cloud disk: https://pan.baidu.com/s/10DkhgxYrU0aem6f_ALYHZQ?pwd=lzms

Download our test datasets on Baidu cloud disk: https://pan.baidu.com/s/1HK1oy4SjZ99N-Dh-8_s0hA?pwd=lzms


### Train

```shell
python train.py -train_batch_size 4 --gpus 0 --type 5
```

### Test

 ```shell
python test.py --gpus 0 --type 5
 ```

### Clone the repo

```sh
git clone https://github.com/thislzm/PSMB-Net.git
```

### Qualitative Results

#### Results on HRSD-DHID remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/DHID.png" style="display: inline-block;" />
</div>

#### Results on HRSD-LHID remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/LHID.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/moderate.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/thick.png" style="display: inline-block;" />
</div>

#### Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/nhhaze.png" style="display: inline-block;" />
</div>

#### Results on RESIDE-Outdoor Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/reside.png" style="display: inline-block;" />
</div>




### Copyright

The project has been licensed by MIT. Please refer to for details. [LICENSE.txt](https://github.com/thislzm/PSMB-Net/LICENSE.txt)

### Thanks


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)


<!-- links -->
[your-project-path]:thislzm/PSMB-Net
[contributors-shield]: https://img.shields.io/github/contributors/thislzm/PSMB-Net.svg?style=flat-square
[contributors-url]: https://github.com/thislzm/PSMB-Net/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/thislzm/PSMB-Net.svg?style=flat-square
[forks-url]: https://github.com/thislzm/PSMB-Net/network/members
[stars-shield]: https://img.shields.io/github/stars/thislzm/PSMB-Net.svg?style=flat-square
[stars-url]: https://github.com/thislzm/PSMB-Net/stargazers
[issues-shield]: https://img.shields.io/github/issues/thislzm/PSMB-Net.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/thislzm/PSMB-Net.svg
[license-shield]: https://img.shields.io/github/license/thislzm/PSMB-Net.svg?style=flat-square
[license-url]: https://github.com/thislzm/PSMB-Net/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian
