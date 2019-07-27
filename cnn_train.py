from cnn_stride import *
# from GP_crop_v3 import *
import logging
import os
import sys
from sklearn.metrics import r2_score


if __name__ == "__main__":
    predict_year = int(sys.argv[-1])
    logging.basicConfig(filename='model.log',level=logging.DEBUG)
    # Create a coordinator
    config = Config()

    # load data to memory
    filename = 'train_data' + '.npz'
    # 13(HeBei) Data is 116, 14(ShanXi) Data is 60, 37(ShanDong) Data is 132,41(HeNan) Data is 105
    cnn_file = os.path.join(config.load_path, filename)
    content = np.load(cnn_file)
    image_all = content['output_image']
    yield_all = content['output_yield']
    year_all = content['output_year']
    locations_all = content['output_locations']
    index_all = content['output_index']

    # split into train and validate
    index_train, index_validate = [], []
    for i in range(year_all.shape[0]):
        if year_all[i] == predict_year:
            # print(i,year_all[0])
            index_validate.append(i)
        else:
            index_train.append(i)
    index_train = np.array(index_train)
    index_validate = np.array(index_validate)
    print ('train size',index_train.shape[0])
    print ('validate size',index_validate.shape[0])

    # calc train image mean (for each band), and then detract (broadcast)
    image_mean=np.mean(image_all[index_train],(0,1,2))
    image_all = image_all - image_mean

    image_validate=image_all[index_validate]
    yield_validate=yield_all[index_validate]

    model= NeuralModel(config,'net')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merge_summary = tf.summary.merge_all()
    board_dir = 'tensorboard/'
    writer = tf.summary.FileWriter(board_dir + '', sess.graph)
    writer.add_graph(sess.graph)

    #parameter initialization
    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []
    train_loss=0
    val_loss=0
    val_prediction = 0
    val_deviation = np.zeros([config.B])

    # #########################
    # block when test
    # add saver
    saver=tf.train.Saver()
    # Restore variables from disk.
    try:
        path = os.path.join(config.save_path,"CNN_model.ckpt")
        saver.restore(sess, path)
        print("Model restored.")
    # Restore log results
        path = os.path.join(config.save_path, str(predict_year) + "result.npz")
        npzfile = np.load(path)
        summary_train_loss = npzfile['summary_train_loss'].tolist()
        summary_eval_loss = npzfile['summary_eval_loss'].tolist()
        summary_RMSE = npzfile['summary_RMSE'].tolist()
        summary_ME = npzfile['summary_ME'].tolist()
        print("Data restored.")
    except:
        print ('No history model found')
    # #########################
    

    RMSE_min = 1000
    try:
        for i in range(config.train_step):

            if i==3500:
                config.lr/=10

            if i==15000:
                config.lr/=10

            if i==30000:
                config.lr/=10

            # try data augmentation while training
            index_train_batch_1 = np.random.choice(index_train,size=config.B)
            index_train_batch_2 = np.random.choice(index_train,size=config.B)
            image_train_batch = (image_all[index_train_batch_1,:,0:config.H,:]+image_all[index_train_batch_1,:,0:config.H,:])/2
            yield_train_batch = (yield_all[index_train_batch_1]+yield_all[index_train_batch_1])/2
            
            index_validate_batch = np.random.choice(index_validate, size=config.B)
            #print(image_train_batch,yield_train_batch)
            _, train_loss = sess.run([model.train_op, model.loss_err], feed_dict={
                model.x:image_train_batch,
                model.y:yield_train_batch,
                model.lr:config.lr,
                model.keep_prob: config.drop_out
                })
            #print(train_loss)
            if i%100 == 0:
                val_loss,fc6,W,B = sess.run([model.loss_err,model.fc6,model.dense_W,model.dense_B], feed_dict={
                    model.x: image_all[index_validate_batch, :, 0:config.H, :],
                    model.y: yield_all[index_validate_batch],
                    model.keep_prob: 1
                })

                print ('predict year'+str(predict_year)+'step'+str(i),model.loss,model.loss_err,train_loss,val_loss,config.lr)
                logging.info('predict year %d step %d %f %f %f',predict_year,i,train_loss,val_loss,config.lr)
            if i%100 == 0:
                # do validation
                pred = []
                real = []
                print('image:',image_validate.shape[0],config.B)
                length = image_validate.shape[0] // config.B
                print('length',length)
                for j in range(image_validate.shape[0] // config.B):
                    real_temp = yield_validate[j * config.B:(j + 1) * config.B]
                    print('real_temp:',real_temp)
                    pred_temp= sess.run(model.logits, feed_dict={
                        model.x: image_validate[j * config.B:(j + 1) * config.B,:,0:config.H,:],
                        model.y: yield_validate[j * config.B:(j + 1) * config.B],
                        model.keep_prob: 1
                        })
                    print('pred_temp',pred_temp)
                    pred.append(pred_temp)
                    real.append(real_temp)
                pred=np.concatenate(pred)
                real=np.concatenate(real)
                RMSE=np.sqrt((np.sum((pred-real)**2)/len(pred)))
                MRE = np.divide(np.sum(np.abs(np.divide(real-pred,real))),len(pred))
                MSE= np.divide(np.sum((pred-real)**2),len(pred))
                MAE= np.divide(np.sum(np.abs(pred-real)),len(pred))
                MAPE=(np.sum(np.abs(pred-real)*100/real))/len(real)
                RSquared = np.subtract(1,np.divide(np.sum(np.power(np.subtract(real,pred),2)),np.sum(np.power(np.subtract(real,np.average(real)),2))))
                REAL_RATIO = 1-(np.sqrt(np.mean((pred-real)**2))/(np.mean(real)))

                if RMSE<RMSE_min:
                    RMSE_min=RMSE
                    # save
                    path = os.path.join(config.save_path,str(predict_year)+str(RMSE_min)+'CNN_model.ckpt')
                    save_path = saver.save(sess, path)
                    print('save in file: %s' % save_path)
                    path = os.path.join(config.save_path, str(predict_year) + str(RMSE_min) + 'result.npz')
                    np.savez(path,
                        summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                        summary_RMSE=summary_RMSE,summary_ME=summary_ME)

                print ('Validation set','RMSE',RMSE,'MRE',MRE,'MSE',MSE,'MAE',MAE,'RSquared',RSquared,'MAPE',MAPE,'REAL_RATIO',REAL_RATIO)
                logging.info('Validation set RMSE %f MRE %f MSE %f MAE %f RSquared %f MAPE %f REAL_RATIO %f',RMSE,MRE,MSE,MAE,RSquared,MAPE,REAL_RATIO)

                summary_train_loss.append(train_loss)
                summary_eval_loss.append(val_loss)
                summary_RMSE.append(RMSE)
                summary_ME.append(RSquared)

            #tensorboard
            if i % 100 == 0:
                s = sess.run(merge_summary,feed_dict={
                    model.x: image_validate[j * config.B:(j + 1) * config.B, :, 0:config.H, :],
                    model.y: yield_validate[j * config.B:(j + 1) * config.B],
                    model.keep_prob: 1
                             })
                writer.add_summary(s,i)

    except KeyboardInterrupt:
        print ('stopped')

    finally:
        # save
        path = os.path.join(config.save_path,'CNN_model.ckpt')
        save_path = saver.save(sess, path)
        print('save in file: %s' % save_path)
        logging.info('save in file: %s' % save_path)

        # save result
        pred_out = []
        real_out = []
        feature_out = []
        year_out = []
        locations_out =[]
        index_out = []
        config.B = 60
        for i in range(image_all.shape[0] // config.B):
            feature,pred = sess.run(
                [model.fc6,model.logits], feed_dict={
                model.x: image_all[i * config.B:(i + 1) * config.B,:,0:config.H,:],
                model.y: yield_all[i * config.B:(i + 1) * config.B],
                model.keep_prob:1
            })
            real = yield_all[i * config.B:(i + 1) * config.B]

            pred_out.append(pred)
            real_out.append(real)
            feature_out.append(feature)
            year_out.append(year_all[i * config.B:(i + 1) * config.B])
            locations_out.append(locations_all[i * config.B:(i + 1) * config.B])
            index_out.append(index_all[i * config.B:(i + 1) * config.B])
            # print i
        weight_out, b_out = sess.run(
            [model.dense_W, model.dense_B], feed_dict={
                model.x: image_all[0 * config.B:(0 + 1) * config.B, :, 0:config.H, :],
                model.y: yield_all[0 * config.B:(0 + 1) * config.B],
                model.keep_prob: 1
            })
        pred_out=np.concatenate(pred_out)
        real_out=np.concatenate(real_out)
        feature_out=np.concatenate(feature_out)
        year_out=np.concatenate(year_out)
        locations_out=np.concatenate(locations_out)
        index_out=np.concatenate(index_out)

        path = os.path.join(config.save_path, str(predict_year) + 'result_prediction.npz')
        np.savez(path,
            pred_out=pred_out,real_out=real_out,feature_out=feature_out,
            year_out=year_out,locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)

        path = os.path.join(config.save_path, str(predict_year) + 'result.npz')
        np.savez(path,
            summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
            summary_RMSE=summary_RMSE,summary_ME=summary_ME)

