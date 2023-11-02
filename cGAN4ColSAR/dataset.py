from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *
from osgeo import gdal, osr
import cv2


def load_image(path):
    img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)  # c,w,h
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

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, mode):
        super(DatasetFromFolder, self).__init__()
        self.gt_path = join(image_dir, "gt_%s" % mode)
        self.sar_path = join(image_dir, "sar_%s" % mode)
        self.image_filenames = [x for x in listdir(self.gt_path) if is_image_file(x)]

    def __getitem__(self, index):
        gt = load_image(join(self.gt_path, self.image_filenames[index]))
        sar = load_image(join(self.sar_path, self.image_filenames[index].split("_")[0] + '_' + self.image_filenames[index].split("_")[1] +
                            '_s1_' + self.image_filenames[index].split("_")[3] + '_' + self.image_filenames[index].split("_")[4]))
        # sar stretch
        sar = sar[0,:,:]
        sar = sar - np.min(sar)
        sar = sar / np.max(sar) * (2**12);  # [0,2**12] 2**12=4096

        # normalization [-1,1]
        sar = (sar - 2048) / 2048
        gt = (gt - 2048) / 2048

        gt = torch.from_numpy(gt).float()  # c,w,h
        sar = torch.from_numpy(sar[np.newaxis, :]).float()  # 1,w,h

        sarcolor_filename = self.image_filenames[index].split("_")[0] + '_' + self.image_filenames[index].split("_")[1] + \
                   '_sarcolor_' + self.image_filenames[index].split("_")[3] + '_' + self.image_filenames[index].split("_")[4]

        return sar, gt, sarcolor_filename

    def __len__(self):
        return len(self.image_filenames)
