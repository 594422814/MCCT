# MCCT(H) tracker
Code for the tracker in the paper **Multi-Cue Correlation Filters for Robust Visual Tracking**, by Ning Wang, Wengang Zhou, Tian Qi and Houqiang Li - appeared at.

###Contacts
For questions about the code or the paper, feel free contact us.

Please cite
```
@InProceedings{Bertinetto_2016_CVPR,
author = {Bertinetto, Luca and Valmadre, Jack and Golodetz, Stuart and Miksik, Ondrej and Torr, Philip H. S.},
title = {Staple: Complementary Learners for Real-Time Tracking},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}
```

###Prerequisites
 - The code is mostly in MATLAB, except the workhorse of `fhog.m`, which is written in C and comes from Piotr Dollar toolbox http://vision.ucsd.edu/~pdollar/toolbox
 - gradientMex and mexResize have been compiled and tested for Ubuntu and Windows 8 (64 bit). You can easily recompile the sources in case of need.
 - To use the webcam mode (`runTracker_webcam`), install MATLAB's webcam support from http://mathworks.com/hardware-support/matlab-webcam.html


###Modes
* `runTracker(sequence, start_frame)` runs the tracker on `sequence` from `start_frame` onwards.
* `runTracker_webcam` starts an interactive webcam demo.
* `runTracker_VOT` and `run_Staple` run the tracker within the benchmarks VOT and OTB respectively.
