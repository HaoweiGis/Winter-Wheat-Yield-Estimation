import numpy as np
import os


Base_dir = r'G:\EstimatedCrop\Data\MODIS_City\History'
npz_file = os.path.join(Base_dir, '2016627.8290195409717result.npz')
r = np.load(npz_file)

pred_out=r["pred_out"]
real_out=r["real_out"]
year_out=r["year_out"]
locations_out=r["locations_out"]


for i in range(len(r["pred_out"])):
    line = str(real_out[i])+','+ str(pred_out[i])+','+str(year_out[i])+','+str(locations_out[i])+'\n'
    open('result2016.csv','a').write(line)