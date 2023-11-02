# A-Benchmarking-Protocol-for-SAR-Colorization-From-Regression-to-Deep-Learning-Approaches
Codes for "A Benchmarking Protocol for SAR Colorization: From Regression to Deep Learning Approaches"

Copyright (c) 2023.
All rights reserved. This work should only be used for nonprofit purposes.

Packages for a  benckmarking protocol for SAR colorization which contains a protocol for generating ground truth, data analysis algorithm, three spectral-based solutions, three deep-learning-based solutions, metrics and performance inspections.

Protocol:
	"filepath_gen.py" is for generating the file path of dataset. And the last null string should be deleted manually.
	"gt_ihs_gen.m" is for generating ground truth image through GIHS algorithm.

Spectral_based_solutions:
	codes for three spectral-based solutions, namely NoColSAR, LR4ColSAR and NL4ColSAR.
	there is also a code for data analysis through scatter plots between SAR band and band of ground truth.

CNN4ColSAR:
	codes for spatial-spectral-based solution CNN4ColSAR.
	a test model file is also contained.
DivColSAR:
	codes for solution DivColSAR.

cGAN4ColSAR:
	codes for solution cGAN4ColSAR.
	a test model file is also contained.

Metrics:
	matlab codes for Q2n.
	python codes for SAM and NRMSE.

Performance_inspection:
	matlab code for data analysis between colorized SAR image and ground truth.
	python code for residual image comparison.

SEN12MS_CR_SARColorData:
	there are 4 train samples and 4 test samples.
	the whole train dataset can be downloaded by the following Baidu Netdisk link:
	https://pan.baidu.com/s/1xHSDNXoQzo5xewMsjQCBrA?pwd=2h7y 
	password：2h7y
	It is noted that the sar_train and opt_train are divided into several parts because of some restrictions, so the users can merge them after decompression. Additionally, sar_train_20, opt_train_20 and gt_train_20 are the trainset for LR4ColSAR and NL4ColSAR.
