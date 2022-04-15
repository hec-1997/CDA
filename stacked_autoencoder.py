import time
import numpy as np
import random
import argparse
from keras.layers import Dense, Input
from keras.models import Sequential, model_from_config, Model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier as GBDT, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn import neighbors
from sklearn import svm
from sklearn import neighbors
from gcforest.gcforest import GCForest
from sklearn.cluster import KMeans
from data_process import *


# auto_encoder
def multiple_layer_autoencoder(X_train, X_test, activation='linear', batch_size=100, nb_epoch=100, last_dim=256):
    nb_hidden_layers = [X_train.shape[1], 1024, 512, last_dim]
    X_train_tmp = np.copy(X_train)
    # X_test_tmp = np.copy(X_test)
    encoders = []
    autoencoders = []
    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
        print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
        # Create AE and training
        # ae = Sequential()
        input_img = Input(shape=(n_in,))
        # encoder = containers.Sequential([Dense(input_dim=n_in, output_dim=n_out, activation=activation)])
        encode = Dense(output_dim=n_out, activation=activation)(input_img)
        # decoder = containers.Sequential([Dense(input_dim=n_out,output_dim=n_in, activation=activation)]) #, activation=activation)])
        decode = Dense(output_dim=n_in, activation=activation)(encode)  # , activation=activation)])
        # ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
        #                   output_reconstruction=True))
        # ae1 = AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True)
        autoencoder = Model(input=input_img, output=decode)
        encoder = Model(input=input_img, output=encode)
        encoded_input = Input(shape=(n_out,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
        autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        autoencoder.fit(X_train_tmp, X_train_tmp, nb_epoch=50, batch_size=20, shuffle=True, validation_data=None,
                        verbose=1)
        encoder.compile(loss='mean_squared_error', optimizer='adam')
        # ae.add(encoder)
        # ae.add(decoder)
        # ae.add(ae1)
        # ae.add(Dropout(0.5))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # ae.compile(loss='mean_squared_error', optimizer='adam')#  adam  'rmsprop')
        # ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=None,show_accuracy=True, verbose=1)
        # Store trainined weight and update training data
        # encoders.append(ae.layers[0].encoder)
        encoders.append(encoder)
        autoencoders.append(autoencoder)
        # ae1.output_reconstruction= False
        X_train_tmp = encoder.predict(X_train_tmp, batch_size=20)
        print(X_train_tmp.shape)
        # X_test_tmp = ae.predict(X_test_tmp)

    # return encoders, X_train_tmp, X_test_tmp
    return encoders, autoencoders


# tuning
def autoencoder_fine_tuning(x_base_train, y_base_train, x_base_test, Y_test, batch_size=20, nb_epoch=50):
    print('autoencode learning')
    last_dim = 256
    encoders1, autoencoders1 = multiple_layer_autoencoder(x_base_train, x_base_test, activation='sigmoid',
                                                          batch_size=batch_size, nb_epoch=nb_epoch, last_dim=last_dim)

    # pdb.set_trace()
    X_train1_tmp_bef = np.copy(x_base_train)
    X_test1_tmp_bef = np.copy(x_base_test)
    for ae in encoders1:
        X_train1_tmp_bef = ae.predict(X_train1_tmp_bef)
        print(X_train1_tmp_bef.shape)
        X_test1_tmp_bef = ae.predict(X_test1_tmp_bef)

    prefilter_train_bef = X_train1_tmp_bef
    prefilter_test_bef = X_test1_tmp_bef

    return prefilter_train_bef, prefilter_test_bef, encoders1
