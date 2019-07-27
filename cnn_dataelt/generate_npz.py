import numpy as np
import math
import os

def batch_generate(yield_index,length):#32,'train'
    output_image = np.zeros([length, 36, 36, 19])
    output_yield = np.zeros([length])

    for n, i in enumerate(yield_index):
        year = str(int(yield_data[i, 0]))
        loc1 = str(int(yield_data[i, 1]))
        loc2 = str(int(yield_data[i, 2]))

        filename = year + '_' + loc1 + '_' + loc2 + '.npy'
        filepath = os.path.join(input_dirs,filename)
        one_image = np.load(filepath)
        one_image[np.isnan(one_image)] = 0
        # print(one_image)
        output_image[n,:] = one_image
        output_yield[n] = yield_data[i, 3]

    return (np.float32(output_image), np.float32(output_yield))


if __name__ == '__main__':

    Base_dir = r'G:\\EstimatedCrop\\Data\\CountyMODIS'
    # Base_dir = r'E:\estimate_corp'
    input_dirs = os.path.join('windows\temp',Base_dir,'hist_data41')
    output_dirs = os.path.join('windows\temp',Base_dir,'cnn_input')

    yield_data = np.genfromtxt('yield_train_41.csv', delimiter=',')
    # print(yield_index)
    yield_index = []
    output_year = []
    output_location = []
    for i in range(yield_data.shape[0]):
        output_year.append(int(yield_data[i,0]))
        output_location.append(int(yield_data[i,2]))
        yield_index.append(i)
    length =yield_data.shape[0]
    output_image,output_yield = batch_generate(yield_index,length)

    cnn_in = os.path.join(output_dirs,'train_data41.npz')
    np.savez(cnn_in,
             output_image = output_image, output_yield = output_yield,
             output_locations = output_location,
             output_index = yield_index,output_year = output_year)



