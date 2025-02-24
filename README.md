# SFRDP-Net
"Spatial-Frequency Residual-guided Dynamic Perceptual Network for remote sensing haze removal"

This work is available at [TGRS2025](https://ieeexplore.ieee.org/document/10892218). Official Pytorch based implementation.

## Abstract
Recently, deep neural networks have been extensively explored in remote sensing image haze removal and achieved remarkable performance. However, most existing haze removal methods fail to effectively leverage the fusion of spatial and frequency information, which is crucial for learning more representative features. Moreover, the prevalent perceptual loss used in dehazing model training overlooks the diversity among perceptual channels, leading to performance degradation. To address these issues, we propose a Spatial-Frequency Residual-guided Dynamic Perceptual Network (SFRDP-Net) for remote sensing image haze removal. Specifically, we first propose a Residual-guided Spatial-Frequency Interaction (RSFI) module, which incorporates a Bidirectional Residual Complementary Mechanism (BRCM) and a Frequency Residual Enhanced Attention (FREA). Both BRCM and FREA exploit spatial-frequency complementarity to guide more effective fusion of spatial and frequency information, thus enhancing feature representation capability and improving haze removal performance. Furthermore, a Dynamic Channel Weighting Perceptual Loss (DCWP-Loss) is developed to impose constraints with varying strengths on different perceptual channels, advancing the reconstruction of high-quality haze-free images. Experiments on challenging benchmark datasets demonstrate our SFRDP-Net outperforms several state-of-the-art haze removal methods.

## Overall architecture
![image]()

## RSFI module
![image]()

## Quantitative results🔥

### COMPARISON OF OUR METHOD AGAINST OTHERS ON THE SATEHAZE1K DATASET. ↑ INDICATES HIGHER IS BETTER, ↓ INDICATES LOWER IS BETTER.

| Method          | Venue&Year  | StateHaze1k-Thin |                 |                 | StateHaze1k-Moderate |                 |                 | StateHaze1k-Thick |                 |                 | StateHaze1k-Average |                 |                 |
|-----------------|-------------|------------------|-----------------|-----------------|----------------------|-----------------|-----------------|-------------------|-----------------|-----------------|---------------------|-----------------|-----------------|
|                 |             | PSNR↑            | SSIM↑           | MSE↓            | PSNR↑                | SSIM↑           | MSE↓            | PSNR↑             | SSIM↑          | MSE↓            | PSNR↑               | SSIM↑          | MSE↓            |
|-----------------|-------------|------------------|-----------------|-----------------|----------------------|-----------------|-----------------|-------------------|-----------------|-----------------|---------------------|-----------------|-----------------|
| DCP [8]         | TPAMI2010   | 13.45            | 0.701           | 0.04452         | 9.780                | 0.592           | 0.10520         | 10.89             | 0.572          | 0.08147         | 11.37               | 0.621          | 0.07706         |
| FFA [9]         | AAAI2020    | 23.75            | 0.903           | 0.00422         | 26.50                | 0.941           | 0.00224         | 22.03             | 0.840          | 0.00627         | 24.09               | 0.894          | 0.00424         |
| DCI-Net [10]    | TGRS2022    | 24.47            | 0.882           | 0.00357         | 25.05                | 0.910           | 0.00313         | 22.73             | 0.806          | 0.00533         | 24.08               | 0.866          | 0.00401         |
| Restormer [11]  | CVPR2022    | 23.08            | 0.912           | 0.00492         | 24.73                | 0.903           | 0.00337         | 19.58             | 0.792          | 0.01102         | 22.46               | 0.869          | 0.00644         |
| FSDGN [12]      | ECCV2022    | 26.18            | 0.914           | 0.00241         | 27.68                | 0.943           | 0.00171         | 23.95             | 0.854          | 0.00663         | 25.94               | 0.904          | 0.00271         |
| Trinity-Net [13]| TGRS2023    | 22.66            | 0.815           | 0.00542         | 25.02                | 0.929           | 0.00315         | 22.05             | 0.825          | 0.00624         | 23.24               | 0.856          | 0.00404         |
| PSMB-Net [14]   | TGRS2023    | 26.75            | 0.928           | 0.00211         | 27.48                | 0.946           | 0.00179         | 25.15             | 0.889          | 0.00056         | 26.46               | 0.921          | 0.00232         |
| C*PNet [15]     | CVPR2023    | 19.62            | 0.880           | 0.01091         | 24.79                | 0.939           | 0.00332         | 16.83             | 0.790          | 0.02075         | 20.41               | 0.870          | 0.01166         |
| FSNet [60]      | TPAMI2024   | 27.33            | 0.929           | 0.00185         | 26.95                | 0.940           | 0.00202         | 25.20             | 0.876          | 0.00302         | 26.49               | 0.915          | 0.00230         |
| DEA-Net [61]    | TIP2024     | 27.11            | 0.931           | 0.00195         | 27.74                | 0.947           | 0.00168         | 24.82             | 0.877          | 0.00330         | 26.56               | 0.918          | 0.00231         |
| MIMO [16]       | TGRS2024    | 27.36            | 0.929           | 0.00184         | 28.37                | 0.957           | 0.00146         | 22.83             | 0.851          | 0.00521         | 26.19               | 0.913          | 0.00284         |
| DDN-SFF [17]    | PR2024      | 26.94            | 0.926           | 0.00203         | 27.93                | 0.943           | 0.00169         | 24.68             | 0.873          | 0.003401        | 26.45               | 0.914          | 0.00237         |
| SFSND [63]      | CVPR2024    | 25.90            | 0.918           | 0.00257         | 26.58                | 0.937           | 0.00220         | 24.35             | 0.868          | 0.00367         | 25.61               | 0.908          | 0.00281         |
| Ours            | -           | 27.88            | 0.931           | 0.00163         | 29.59                | 0.958           | 0.00110         | 25.99             | 0.890          | 0.00252         | 27.82               | 0.926          | 0.00175         |

### COMPARISON RESULTS OF OUR METHOD WITH OTHER ADVANCED METHODS ON THE RICE AND DHID DATASETS, INCLUDING PARAMETERS. ↑: LARGER IS BETTER. ↓: SMALLER IS BETTER.

| Method          | Venue&Year  | RICE1         |                 |                 | RICE2         |                 |                 | DHID          |                 |                 | Overhead       |               |
|-----------------|-------------|---------------|-----------------|-----------------|---------------|-----------------|-----------------|---------------|-----------------|-----------------|----------------|---------------|
|                 |             | **PSNR↑**     | **SSIM↑**       | **MSE↓**        | **PSNR↑**     | **SSIM↑**       | **MSE↓**        | **PSNR↑**     | **SSIM↑**       | **MSE↓**        | **#Param↓**   | **FLOPs↓**    |
|-----------------|-------------|---------------|-----------------|-----------------|---------------|-----------------|-----------------|---------------|-----------------|-----------------|----------------|---------------|
| DCP [8]         | TPAMI2010   | 12.39         | 0.628           | 0.05768         | 12.47         | 0.535           | 0.05662         | 12.96         | 0.517           | 0.05058         | -              | 0.6G          |
| FFA [9]         | AAAI2020    | 35.01         | 0.951           | 0.00032         | 34.82         | 0.894           | 0.00033         | 27.07         | 0.863           | 0.00196         | 4.69M          | 287.53G       |
| DCI-Net [10]    | TGRS2022    | 35.03         | 0.960           | 0.00031         | 32.14         | 0.886           | 0.00061         | 28.17         | 0.892           | 0.00152         | 26.51M         | 26.86G        |
| Restormer [11]  | CVPR2022    | 27.92         | 0.931           | 0.00161         | 24.61         | 0.801           | 0.00346         | 25.61         | 0.853           | 0.00275         | 26.10M         | 141.0G        |
| FSDGN [12]      | ECCV2022    | 33.09         | 0.913           | 0.00049         | 33.56         | 0.893           | 0.00044         | 27.85         | 0.876           | 0.00164         | 2.73M          | 19.60G        |
| Trinity-Net [13]| TGRS2023    | 27.30         | 0.913           | 0.00186         | 24.71         | 0.794           | 0.00338         | 26.97         | 0.871           | 0.00201         | 20.14M         | 30.78G        |
| PSMB-Net [14]   | TGRS2023    | 34.68         | 0.963           | 0.00034         | 33.96         | 0.893           | 0.00040         | 28.01         | 0.888           | 0.00158         | 12.49M         | 98.19G        |
| C<sup>2</sup>PNet [15] | CVPR2023 | 28.64 | 0.911           | 0.00137         | 25.46         | 0.811           | 0.00284         | 27.85         | 0.881           | 0.00164         | 7.17M          | 460.95G       |
| FSNet [60]      | TPAMI2024   | 37.06         | 0.965           | 0.00020         | 34.83         | 0.907           | 0.00033         | 28.51         | 0.907           | 0.00141         | 13.28M         | 111.14G       |
| DEA-Net [61]    | TIP2024     | 36.12         | 0.963           | 0.00024         | 34.13         | 0.889           | 0.00039         | 28.12         | 0.906           | 0.00154         | 3.65M          | 32.23G        |
| MIMO [16]       | TGRS2024    | 36.01         | 0.962           | 0.00025         | 34.28         | 0.904           | 0.00037         | 28.65         | 0.899           | 0.00136         | 15.28M         | 46.44G        |
| DNN-SFF [17]    | PR2024      | 35.41         | 0.944           | 0.00029         | 34.64         | 0.903           | 0.00034         | 28.09         | 0.894           | 0.00155         | 29.5M          | 65.15G        |
| SFSNID [63]     | CVPR2024    | 35.93         | 0.965           | 0.00026         | 34.68         | 0.908           | 0.00034         | 28.12         | 0.903           | 0.00154         | 8.35M          | 69.36G        |
| Ours            | -           | 37.38         | 0.965           | 0.00018         | 35.45         | 0.910           | 0.00028         | 28.89         | 0.907           | 0.00129         | 5.08M          | 38.09G        |

## Qualitative results🔥

#### Results on Statehaze1k-Thin testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-moderate testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thick testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Rice1 testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Rice2 testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on DHID testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

### Pretrained Weights✨ and Dataset🤗

Download our model weights on Baidu cloud disk: https://pan.baidu.com/s/10DkhgxYrU0aem6f_ALYHZQ?pwd=lzms

Download our test datasets on Baidu cloud disk: https://pan.baidu.com/s/1HK1oy4SjZ99N-Dh-8_s0hA?pwd=lzms

You can load these models to generate images via the codes in [test.py](test.py).

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

```

```

```

### Thanks

Center for Advanced Computing, School of Computer Science, China Three Gorges University
