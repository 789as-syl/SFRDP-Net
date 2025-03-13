# SFRDP-Net
"Spatial-Frequency Residual-guided Dynamic Perceptual Network for remote sensing haze removal"

This work is available at [TGRS2025](https://ieeexplore.ieee.org/document/10892218). Official Pytorch based implementation.

## Abstract
Recently, deep neural networks have been extensively explored in remote sensing image haze removal and achieved remarkable performance. However, most existing haze removal methods fail to effectively leverage the fusion of spatial and frequency information, which is crucial for learning more representative features. Moreover, the prevalent perceptual loss used in dehazing model training overlooks the diversity among perceptual channels, leading to performance degradation. To address these issues, we propose a Spatial-Frequency Residual-guided Dynamic Perceptual Network (SFRDP-Net) for remote sensing image haze removal. Specifically, we first propose a Residual-guided Spatial-Frequency Interaction (RSFI) module, which incorporates a Bidirectional Residual Complementary Mechanism (BRCM) and a Frequency Residual Enhanced Attention (FREA). Both BRCM and FREA exploit spatial-frequency complementarity to guide more effective fusion of spatial and frequency information, thus enhancing feature representation capability and improving haze removal performance. Furthermore, a Dynamic Channel Weighting Perceptual Loss (DCWP-Loss) is developed to impose constraints with varying strengths on different perceptual channels, advancing the reconstruction of high-quality haze-free images. Experiments on challenging benchmark datasets demonstrate our SFRDP-Net outperforms several state-of-the-art haze removal methods.

## Overall architecture
![image](/images/net.png)

## RSFI module
![image](/images/RSFI.png)

## Quantitative results🔥

### COMPARISON OF OUR METHOD AGAINST OTHERS ON THE SATEHAZE1K DATASET. ↑ INDICATES HIGHER IS BETTER, ↓ INDICATES LOWER IS BETTER.
<div style="text-align: center">
<img alt="" src="/images/Table1.png" style="display: inline-block;" />
</div>

### COMPARISON RESULTS OF OUR METHOD WITH OTHER ADVANCED METHODS ON THE RICE AND DHID DATASETS, INCLUDING PARAMETERS. ↑: LARGER IS BETTER. ↓: SMALLER IS BETTER.
<div style="text-align: center">
<img alt="" src="/images/Table2.png" style="display: inline-block;" />
</div>

## Qualitative results🔥

#### Results on Statehaze1k-Thin testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-moderate testing images
<div style="text-align: center">
<img alt="" src="/images/moderate.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thick testing images
<div style="text-align: center">
<img alt="" src="/images/thick.png" style="display: inline-block;" />
</div>

#### Results on Rice1 testing images
<div style="text-align: center">
<img alt="" src="/images/rice1.png" style="display: inline-block;" />
</div>

#### Results on Rice2 testing images
<div style="text-align: center">
<img alt="" src="/images/rice2.png" style="display: inline-block;" />
</div>

#### Results on DHID testing images
<div style="text-align: center">
<img alt="" src="/images/DHID.png" style="display: inline-block;" />
</div>

## Ablation results🔥

### THE PSNR, SSIM AND MSE AT EACH STAGE OF THE ABLATION EXPERIMENT ARE CONDUCTED ON THE STATEHAZE1K DATASET. ↑: LARGER IS BETTER. ↓: SMALLER IS BETTER.

<div style="text-align: center">
<img alt="" src="/images/Table3.png" style="display: inline-block;" />
</div>

### EVALUATION OF APPLYING OUR DYNAMIC CHANNEL WEIGHTING PERCEPTUAL LOSS TO SEVERAL STATE-OF-THE-ART HAZE REMOVAL METHODS

<div style="text-align: center">
<img alt="" src="/images/Table4.png" style="display: inline-block;" />
</div>

#### Results on DHID testing images

### Pretrained Weights✨ and Dataset🤗

Download our model weights on Baidu cloud disk: 

Download our test datasets on Baidu cloud disk: 

You can load these models to generate images via the codes in [test.py](SFRDP-Net/test.py).

### Filetree

```
├── README.md
├── /images/
├── /SFRDP-Net/
|  ├── train.py
|  ├── test.py
|  ├── Net.py
|  ├── config.json
|  ├── data_utils.py
|  ├── loss.py
|  ├── metrics.py
|  ├── model.py
|  ├── option.py
│  ├── /pytorch_msssim/
│  │  ├── __init__.py

```

### Train

```shell
python train.py 
```

### Test

 ```shell
python test.py 
 ```

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@ARTICLE{10892218,
  author={Sun, Hang and Yao, Zhaoru and Du, Bo and Wan, Jun and Ren, Dong and Tong, Lyuyang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Spatial–Frequency Residual-Guided Dynamic Perceptual Network for Remote Sensing Image Haze Removal}, 
  year={2025},
  volume={63},
  number={},
  pages={1-16},
  keywords={Feature extraction;Atmospheric modeling;Image reconstruction;Training;Remote sensing;Scattering;Logic gates;Image restoration;Frequency-domain analysis;Degradation;Dynamic channel weighting;haze removal;remote sensing image;residual information;spatial-frequency complementarity},
  doi={10.1109/TGRS.2025.3543728}}

```

```
H. Sun, Z. Yao, B. Du, J. Wan, D. Ren and L. Tong, "Spatial–Frequency Residual-Guided Dynamic Perceptual Network for Remote Sensing Image Haze Removal," in IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1-16, 2025, Art no. 5612416, doi: 10.1109/TGRS.2025.3543728.
keywords: {Feature extraction;Atmospheric modeling;Image reconstruction;Training;Remote sensing;Scattering;Logic gates;Image restoration;Frequency-domain analysis;Degradation;Dynamic channel weighting;haze removal;remote sensing image;residual information;spatial-frequency complementarity},
```

### Thanks

Center for Advanced Computing, School of Computer Science, China Three Gorges University
