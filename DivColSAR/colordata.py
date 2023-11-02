import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from osgeo import gdal, osr
from os.path import join
from os import listdir


def is_image_file(filename):
  return any(filename.endswith(extension) for extension in [".tif", ".png", ".jpg", ".npz"])

def load_image(path):
  img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.float32)
  return img

def save_image(path, array, bandSize):
  rasterOrigin = (-123.25745, 45.43013)
  pixelWidth = 2.4
  pixelHeight = 2.4

  if (bandSize != 1):
    cols = array.shape[2]
    rows = array.shape[1]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')

    outRaster = driver.Create(path, cols, rows, bandSize, gdal.GDT_UInt16)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    for i in range(1, bandSize + 1):
      outband = outRaster.GetRasterBand(i)
      outband.WriteArray(array[i - 1, :, :])
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
  elif (bandSize == 1):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    driver = gdal.GetDriverByName('GTiff')

    outRaster = driver.Create(path, cols, rows, 1, gdal.GDT_UInt16)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array[:, :])

class colordata(Dataset):

  def __init__(self, out_directory, image_dir, shape=(64, 64), outshape=(256, 256), mode='train'):
    # super(DatasetFromFolder, self).__init__()
    self.gt_path = join(image_dir, "gt_%s_3bands" % mode)
    self.sar_path = join(image_dir, "sar_%s_2bands" % mode)
    self.feat_path = join(image_dir, "DivColorData/sen12ms_cr_feats/%s" % mode)

    self.image_filenames = [x for x in listdir(self.gt_path) if is_image_file(x)]
    self.feat_filenames = [x for x in listdir(self.feat_path) if is_image_file(x)]

    self.shape = shape
    self.outshape = outshape
    self.out_directory = out_directory

    self.img_num = len(self.image_filenames)


  def __len__(self):
    return len(self.image_filenames)
 
  def __getitem__(self, index):
    greyfeats = np.zeros((512, 28, 28), dtype='f')

    gt = load_image(join(self.gt_path, self.image_filenames[index]))  # 这里面的图片是3*256*256的彩色图片 [3,256,256]
    sar = load_image(join(self.sar_path, self.image_filenames[index].split("_")[0] + '_' + self.image_filenames[index].split("_")[1] +
           '_s1_' + self.image_filenames[index].split("_")[3] + '_' + self.image_filenames[index].split("_")[4]))

    # sar stretch
    sar = sar[0, :, :]
    sar = sar - np.min(sar)
    # sar = sar / np.max(sar) * (2 ** 12);  # [0,2**12] 2**12=4096
    sar = sar / np.max(sar);  # [0,1]
    sar = sar[np.newaxis, :]  # [1,256,256]

    # gt = torch.from_numpy(gt).float()  # c,w,h
    # sar = torch.from_numpy(sar[np.newaxis, :]).float()  # 1,w,h

    gt = gt/(2**12)

    featobj = np.load(join(self.feat_path,self.feat_filenames[index]))
    greyfeats[:,:,:] = featobj['arr_0']   # [1,512,28,28], MDN input

    greyfeats = greyfeats - np.min(greyfeats)
    greyfeats = greyfeats / np.max(greyfeats)

    sarcolor_filename = self.image_filenames[index].split("_")[0] + '_' + self.image_filenames[index].split("_")[1] + \
                        '_sarcolor_' + self.image_filenames[index].split("_")[3] + '_' + \
                        self.image_filenames[index].split("_")[4]

    return gt, sar, greyfeats, sarcolor_filename


  def tiledoutput(self, prefix, net_op, batchsize):
    for i in range(batchsize):
      out_fn = '%s/%s' % (self.out_directory, prefix[i])
      save_image(out_fn, net_op[i,:,:,:], 3)


