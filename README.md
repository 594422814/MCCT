# Multi-Cue Correlation Tracker 
Code for tracker in the paper **Multi-Cue Correlation Filters for Robust Visual Tracking**, by Ning Wang, Wengang Zhou et. al - to appear in CVPR 2018. 

In this work, we propose to utilize multiple weak experts for online tracking. Our efficient framework achieves state-of-the-art performance just using simple and standard DCFs! 

### Contacts
For questions about the code or paper, please feel free to contact me: wn6149@mail.ustc.edu.cn

### Citing
If you find MCCT useful in your research, please consider citing:
```
@InProceedings{NingCVPR2018,  
	Title                    = {Multi-Cue Correlation Filters for Robust Visual Tracking},  
	Author                   = {Ning Wang, Wengang Zhou, Qi Tian, Richang Hong, Meng Wang, Houqiang Li},  
	Booktitle                = {CVPR},  
	Year                     = {2018}  
}
```
### Prerequisites
For the MCCT-H (Hand-crafted features only) tracker, just start Matlab and run the ```runTracker.m```. To run the MCCT tracker with deep features, please download the VGG-19 and compile the Matconvnet following the description in README (in ```MCCT/model/```).
 - The VGG-19 model is available at http://www.vlfeat.org/matconvnet/pretrained/.
 - The Matconvnet is available at https://github.com/vlfeat/matconvnet.
 - The code is mostly in MATLAB, except the workhorse of `fhog.m`, which is written in C and comes from Piotr Dollar toolbox http://vision.ucsd.edu/~pdollar/toolbox
 - gradientMex and mexResize have been compiled and tested for Ubuntu and Windows 8 (64 bit). You can easily recompile the sources in case of need.

### Acknowledgments
Some codes of this work are adopted from previous trackers (Staple, HCF).
- L. Bertinetto, J. Valmadre, S. Golodetz, O. Miksik, and P. Torr. Staple: Complementary learners for real-time tracking. In CVPR, 2016.
- C. Ma, J.-B. Huang, X. Yang, and M.-H. Yang. Hierarchical convolutional features for visual tracking. In ICCV, 2015.

