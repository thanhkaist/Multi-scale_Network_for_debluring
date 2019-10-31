python test.py one_scale --pretrained_model one_scale1/Net1/model/model_best.pt --gpu 0
python test.py one_scale_lsc --pretrained_model one_scale_lsc1/Net1/model/model_best.pt --gpu 0
python test.py multi_scale --pretrained_model multi_scale1/Net1/model/model_best.pt --gpu 0
python test.py multi_scale_lsc --pretrained_model multi_scale_lsc1/Net1/model/model_best.pt --gpu 0
python test.py multi_scale_lsc --saveDir GoPro_best --pretrained_model multi_scale_lsc1000/Net1/model/model_best.pt --gpu 0
