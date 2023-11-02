from __future__ import print_function
import os
import time
from datetime import datetime
import numpy as np
import cv2

# create directory
gt_dir = './gt/'
nocolsar_dir = './nocolsar/'
lr4colsar_dir = './lr4colsar/'
nl4colsar_dir = './nl4colsar/'
cnn4colsar_dir = './cnn4colsar/'
can4colsar_dir = './can4colsar/'
nocol_diff_dir = './result_nocol_diff/'
gt_diff_dir = './result_gt_diff/'

name = ['_119_495.tif', '_120_408.tif','_121_560.tif','_8_251.tif']

if not os.path.exists(gt_diff_dir):
    os.makedirs(gt_diff_dir)
if not os.path.exists(nocol_diff_dir):
    os.makedirs(nocol_diff_dir)

### (X - nocol) difference
nocolsar_nocol_diff_sum, lr4colsar_nocol_diff_sum, nl4colsar_nocol_diff_sum, \
cnn4colsar_nocol_diff_sum,can4colsar_nocol_diff_sum,gt_nocol_diff_sum = 0.0,0.0,0.0,0.0,0.0,0.0

for i in range(len(name)):
    nocolsar_nocol_diff = abs(cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])) - cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])))
    lr4colsar_nocol_diff = abs(cv2.imread('%s%s%s' % (lr4colsar_dir,'lr4colsar',name[i])) - cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])))
    nl4colsar_nocol_diff = abs(cv2.imread('%s%s%s' % (nl4colsar_dir,'nl4colsar',name[i])) - cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])))
    cnn4colsar_nocol_diff = abs(cv2.imread('%s%s%s' % (cnn4colsar_dir,'cnn4colsar',name[i])) - cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])))
    can4colsar_nocol_diff = abs(cv2.imread('%s%s%s' % (can4colsar_dir,'can4colsar',name[i])) - cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])))
    gt_nocol_diff = abs(cv2.imread('%s%s%s' % (gt_dir,'gt',name[i])) - cv2.imread('%s%s%s' % (nocolsar_dir,'nocolsar',name[i])))

    cv2.imwrite('%s%s%s' % (nocol_diff_dir, 'nocolsar_nocol_diff', name[i]), nocolsar_nocol_diff)
    cv2.imwrite('%s%s%s' % (nocol_diff_dir, 'lr4colsar_nocol_diff', name[i]), lr4colsar_nocol_diff)
    cv2.imwrite('%s%s%s' % (nocol_diff_dir, 'nl4colsar_nocol_diff', name[i]), nl4colsar_nocol_diff)
    cv2.imwrite('%s%s%s' % (nocol_diff_dir, 'cnn4colsar_nocol_diff', name[i]), cnn4colsar_nocol_diff)
    cv2.imwrite('%s%s%s' % (nocol_diff_dir, 'can4colsar_nocol_diff', name[i]), can4colsar_nocol_diff)
    cv2.imwrite('%s%s%s' % (nocol_diff_dir, 'gt_nocol_diff', name[i]), gt_nocol_diff)

    nocolsar_nocol_diff_sum = nocolsar_nocol_diff_sum + np.mean(nocolsar_nocol_diff)
    lr4colsar_nocol_diff_sum = lr4colsar_nocol_diff_sum + np.mean(lr4colsar_nocol_diff)
    nl4colsar_nocol_diff_sum = nl4colsar_nocol_diff_sum + np.mean(nl4colsar_nocol_diff)
    cnn4colsar_nocol_diff_sum = cnn4colsar_nocol_diff_sum + np.mean(cnn4colsar_nocol_diff)
    can4colsar_nocol_diff_sum = can4colsar_nocol_diff_sum + np.mean(can4colsar_nocol_diff)
    gt_nocol_diff_sum = gt_nocol_diff_sum + np.mean(gt_nocol_diff)

print('nocolsar_nocol_diff_mean_value',nocolsar_nocol_diff_sum / len(name))
print('lr4colsar_nocol_diff_mean_value',lr4colsar_nocol_diff_sum / len(name))
print('nl4colsar_nocol_diff_mean_value',nl4colsar_nocol_diff_sum / len(name))
print('cnn4colsar_nocol_diff_mean_value',cnn4colsar_nocol_diff_sum / len(name))
print('can4colsar_nocol_diff_mean_value',can4colsar_nocol_diff_sum / len(name))
print('gt_nocol_diff_mean_value',gt_nocol_diff_sum / len(name))
