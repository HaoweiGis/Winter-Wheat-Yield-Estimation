import os 
import tensorflow as tf
import numpy as np
from cnn_stride import *

config = Config()
filename = 'train_data' + '.npz'
Base_dir = r'G:\EstimatedCrop\Data\MODIS_City'
Base_dir1 = r'G:\EstimatedCrop\Data\MODIS_City\History'
cnn_file = os.path.join(Base_dir, 'cnn_input')
content = np.load(cnn_file)
image_all = content['output_image']
# print(image_all)
yield_all = content['output_yield']
year_all = content['output_year']
locations_all = content['output_locations']
index_all = content['output_index']

model= NeuralModel(config,'net')
metapath = os.path.join(Base_dir1, '2016627.8290195409717CNN_model.ckpt')
# pointpath="./cnn_output"
sess = tf.Session()
saver = tf.train.Saver()
# saver = tf.train.import_meta_graph(metapath)
saver.restore(sess, metapath)

# save result
pred_out = []
real_out = []
feature_out = []
year_out = []
locations_out =[]
index_out = []
batch = 36
for i in range(image_all.shape[0] // batch):
    feature,pred = sess.run(
        [model.fc6,model.logits], feed_dict={
        model.x: image_all[i * batch:(i + 1) * batch,:,0:config.H,:],
        model.y: yield_all[i * batch:(i + 1) * batch],
        model.keep_prob:1
    })
    real = yield_all[i * batch:(i + 1) * batch]

    pred_out.append(pred)
    real_out.append(real)
    feature_out.append(feature)
    year_out.append(year_all[i * batch:(i + 1) * batch])
    locations_out.append(locations_all[i * batch:(i + 1) * batch])
    index_out.append(index_all[i * batch:(i + 1) * batch])
    # print i

pred_out=np.concatenate(pred_out)
real_out=np.concatenate(real_out)
feature_out=np.concatenate(feature_out)
year_out=np.concatenate(year_out)
locations_out=np.concatenate(locations_out)
index_out=np.concatenate(index_out)

path = 'result_prediction.npz'
np.savez(path,
    pred_out=pred_out,real_out=real_out,feature_out=feature_out,
    year_out=year_out,locations_out=locations_out,index_out=index_out)
