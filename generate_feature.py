from keras import backend as K
from keras.models import load_model
from keras.preprocessing import sequence
import gzip
import pickle
import numpy as np
def load_data(dataset):
    '''
    读取数据，暂不设验证集
    :param dataset:
    :return:
    '''
    max_features = 2298
    TRAIN_SET = max_features
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    test_x = data_x[TRAIN_SET:]
    test_y = data_y[TRAIN_SET:]
    return train_x, train_y, test_x, test_y

# load data
filename_data = 'data/smp_10Gword_50dim.pretrain.pkl.gz'
train_x, train_y, test_x, test_y = load_data(filename_data)
maxlen = 20  # cut texts after this number of words (among top max_features most common words)
# 限定最大词数
len_wv = 50
# Memory 足够时用
train_x = sequence.pad_sequences(train_x, maxlen=maxlen, dtype="float64")
train_x = train_x.reshape((len(train_x), maxlen, len_wv))
# load model
model = load_model("model_best/309.5-maxf_smp_31train_31develop_weights.epoch_10-val_acc_0.94.hdf5")

# with a Sequential model
get_1rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])

layer_output = get_1rd_layer_output([train_x])[0]
np.save("dnn_feature/feature.npy", layer_output)
