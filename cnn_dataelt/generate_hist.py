import numpy as np
import scipy.io as io
import math
import os
import skimage.io
import pandas as pd
import gdal
import sys

def divide_image(img,first,step,num):
    image_list=[]
    for i in range(0,num-1):
        img_band = img[:, :, first:first+step]
        image_list.append(img_band)
        first+=step
    image_list.append(img[:, :, first:])
    return image_list

def read_image(MODIS_path):
    image = None
    for _, _,group_file in os.walk(MODIS_path):
        # print(group_file)
        for sigle_file in group_file:
            print(sigle_file)
            sigle_file = os.path.join('windows\temp',MODIS_path, sigle_file)
            sigle_img = np.transpose(np.array(gdal.Open(sigle_file).ReadAsArray(), dtype='int16'), axes=(1, 2, 0))
            if image is None:
                image = sigle_img
            else:
                image = np.concatenate((image, sigle_img), axis=2)
        # print('MODIS_shape',image.shape)
    return image

def filter_size(image_temp,size):
    return image_temp[0:size, 0:size, :].astype(np.float32)

def filter_abnormal(image_temp,min,max):
    image_temp[image_temp<min]=min
    image_temp[image_temp>max]=max
    return image_temp

def calc_histogram(image_temp, bins, times, bands):#(image_temp,36,36,9)
    hist = np.zeros([bins, times, bands])
    # print(image_temp.shape)
    for i in range(image_temp.shape[2]):
        zhibiao = i%19
        # print(zhibiao)
        if zhibiao <7:     
            bin_seq = np.linspace(1,5000,37)
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,5001)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 7:
            sigle_normal = filter_abnormal(image_temp[:, :, i],12999,16001)
            bin_seq = np.linspace(13000,16000,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 8:
            sigle_normal = filter_abnormal(image_temp[:, :, i],12999,15001)
            bin_seq = np.linspace(13000,15000,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 9:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,301)
            bin_seq = np.linspace(1,300,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 10 or zhibiao == 11:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,801)
            bin_seq = np.linspace(1,800,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 12:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,2001)
            bin_seq = np.linspace(1,2000,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 13:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,201)
            bin_seq = np.linspace(1,200,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 14:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,351)
            bin_seq = np.linspace(1,350,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 15 or zhibiao == 16:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,9001)
            bin_seq = np.linspace(1,9000,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 17:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,76)
            bin_seq = np.linspace(1,75,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
        elif zhibiao == 18:
            sigle_normal = filter_abnormal(image_temp[:, :, i],0,31)
            bin_seq = np.linspace(1,30,37)
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # density[np.isnan(density)] = 0
            den_sum = np.sum(density)
            percent = density /den_sum
            hist[:, i // bands, i % bands] = percent
    # print(image_temp.shape[2])
    return hist

def preprocess_save_data_parallel(filepath):

    MODIS_dir=os.path.join('windows\temp',Base_dir,'image_data41')

    img_output_dir=os.path.join('windows\temp',Base_dir,'hist_data41')

    data_yield = np.genfromtxt('yield_train.csv', delimiter=',', dtype=float)

    MODIS_path = os.path.join('windows\temp',MODIS_dir,filepath)

    raw = filepath.replace('_',' ').replace('.',' ').split()
    loc1 = int(raw[0])
    loc2 = int(raw[1])
    MODIS_img = read_image(MODIS_path)

    MODIS_img_list=divide_image(MODIS_img, 0, 36 * 19, year_num)
    hist_list = []
    for year in range(len(MODIS_img_list)):
        sigle_hist = MODIS_img_list[year]
        sigle_hist = calc_histogram(sigle_hist,36,36,19)
        hist_list.append(sigle_hist)

    for i in range(0, year_num):
        year = i+year_start
        key = np.array([year,loc1,loc2])
        if np.sum(np.all(data_yield[:,0:3] == key, axis=1))>0:
            filename=str(year)+'_'+str(loc1)+'_'+str(loc2)+'.npy'
            filename=os.path.join('windows\temp',img_output_dir,filename)
            np.save(filename,hist_list[i])
            print (filename,':written ')

if __name__ == "__main__":
    # # save data
    Base_dir=r'G:\\EstimatedCrop\\Data\\CountyMODIS'
    MODIS_dir=os.path.join('windows\temp',Base_dir,'image_data41')
    year_start = int(sys.argv[-2])
    year_num = int(sys.argv[-1])
    # print('shijian:',year_start,year_num)
    for _, dirs, files in os.walk(MODIS_dir):
        print(dirs)
        for filepath in dirs:
            print('this1'+filepath)
            preprocess_save_data_parallel(filepath)

