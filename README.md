# Residual Inpainting Using Selective Free-Form Attention
## Prerequisites
- Python 3.6
- PyTorch 1.2
- NVIDIA GPU + CUDA cuDNN
- Some other frequently-used python packages, like opencv, numpy, imageio, etc.

## Datasets prepration
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris Street View](https://github.com/pathak22/context-encoder) and [Places2](http://places2.csail.mit.edu/) datasets. 
The irregular mask dataset is available from [here](http://masc.cs.gmu.edu/wiki/partialconv).
After Downloading images and masks, create .filst file containing the dataset path in `./datasets/` (some examples have been given, refer to so).

## Training
To continue training, first download pretrained models from my [OneDrive](https://tjueducn-my.sharepoint.com/:f:/g/personal/yangshiyuan_tju_edu_cn/EgbzPRGqkYVGg9GJUA8E06EBUyb3RQ-CbJAbXROHSWGolA?e=1bg5ev), and place .pth files in the `./checkpoints` directory.

Please edit the config file `./config.yml` for your training setting.
The options are all included in this file, see comments for the explanations. 

Once you've set up, run the `./train.py` script to launch the training.
```shell script
python train.py
```

## Testing
Please download pretrained models from my [OneDrive](https://tjueducn-my.sharepoint.com/:f:/g/personal/yangshiyuan_tju_edu_cn/EgbzPRGqkYVGg9GJUA8E06EBUyb3RQ-CbJAbXROHSWGolA?e=1bg5ev), and place .pth files in the `./checkpoints` directory.

Use `.test.py` for testing, you can directly run this script without any arguments:
```shell script
python test.py
```
By default, this will inpaint the example images under the `examples/celeba/images` with the masks `examples/celeba/masks`. The output results will be saved in `./results`. 

Note that please use the original image rather than masked image as the input, our model will do the masking operation. Using masked image as input will introduce corss-color artifact since our model contains downsampling process. This issue will be fixed in the future.

For customized path, here are some args：
- `--G1` path to generator 1
- `--G2` path to generator 2
- `--input` path to input images
- `--mask` path to masks
- `--output` path to results directory

Alternatively, you can also edit these options in the config file `./config.yml`.

## Acknowledgement
This project is modified based on the [Edge-Connect](https://github.com/knazeri/edge-connect) Model proposed by Nazeri et al.

