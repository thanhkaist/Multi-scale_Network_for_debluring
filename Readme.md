EE838 HOME WORK 2
======================
### Prerequisite

Dowload data set from the link: 
Unzip GoPro dataset to **data** folder such that we have
- data/train : train data
- data/test : test data

Set up environment:

```conda create -s deblur python=3.6```\
```conda activate deblur```\
```pip install -r requirement.txt```

### How to train 
Train the network by run corresponding command below:

One scale

```./one_scale_no_lsc.sh``` 

One scale with long skip connection

```./one_scale__lsc.sh```

Multi scale 

```./multi_scale_no_lsc.sh```

Multi scale with long skip connection 

```./multi_scale_with_lsc.sh```

### How to test 
We provide pretrain model at url:

upzip the model to src folder and run:

```./test_model.sh```

In case that you want to test your train model, read the test_model.sh and modify the pretrained_model path.

### PSNR, SSIM, MS-SSIM
I used **SKIMAGE** library for calculate PSNR and SSIM 

```python
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
```

For MS-SSIM, I used **Tensorflow** code which is available at: https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
The code is hard copy to **utils.py**, so we don't need to worry about the dependency. 

 
### Result

