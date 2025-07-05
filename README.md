# An Arbitrary scale super resolution methods for stereo-vision
### Title: StereoINR: Cross-View Geometry Consistent Stereo Super Resolution with Implicit Neural Representation

### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks.
    
    conda create -n stereoinr python==3.9
    pip install -r  requirements.txt  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    cd  StereoINR
    python setup.py develop
            
## 1. Quick Test 
#### 1.1 Download the pretrained model to the dir of 'experiments/pretrained_models'.
#####
   *pretrained model can be download at ,
       
#### 1.2 Modify the dataroot_lq: in  'options/test/ASSR'
        test_StereoINR_scale.yml

#### 1.3 Run the test scripts 
        python basicsr/test.py -opt options/test/ASSR/test_StereoINR_scale.yml
#### 1.4 The final results are in 'results'

If StereoINR is helpful, please help to ⭐ the repo.

### Contact

If you have any questions, please contact liuyiwhu28@whu.edu.com
 

    
    
    
    
        
