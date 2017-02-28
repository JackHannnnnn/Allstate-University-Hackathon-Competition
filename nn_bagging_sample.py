#!/usr/bin/env python -W ignore::DeprecationWarning

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
from sklearn.cross_validation import KFold
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from datetime import datetime
import argparse
from sklearn.metrics import log_loss


## Batch generators 
def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :]  
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0



def nn_fit_predict(args):
    layer1 = args['layer1']
    layer2 = args['layer2']

    ## neural net
    def nn_model():
        model = Sequential()

        model.add(Dense(layer1, input_dim = train_x.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(args['layer1_dp']))

        model.add(Dense(layer2, init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(args['layer2_dp']))

        if 'layer3' in args:
            print 'layer3'
            print '\n\n\n'
            model.add(Dense(args['layer3'], init = 'he_normal'))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(args['layer3_dp']))
        
        model.add(Dense(10, init = 'he_normal', activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta')
        return model


    start = datetime.now()
    print('\nStart reading data')
    print start

    feature_type = args['feature_type']
    train_x = pd.read_csv('data/' + feature_type + '_dtrain.csv', header=None)
    test_x = pd.read_csv('data/' + feature_type + '_dtest.csv', header=None)
    train_y = pd.read_csv('data/' + 'ytrain.txt', header=None).values[:, 0]
    train_y = pd.get_dummies(train_y).values

    ntrain = train_x.shape[0]
    ntest = test_x.shape[0]

    print 'Dim of the training set: ', train_x.shape
    print 'Dim of the test set: ', test_x.shape


    num_class = 10
    NFOLDS = 10
    nepochs = 80
    cv_sum = 0

    oof_train = np.zeros((ntrain, num_class))
    oof_test = np.zeros((ntest, num_class))
    oof_test_skf = np.zeros((ntest, num_class))


    print 'Start training'
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=2016)
    for i, (train_index, test_index) in enumerate(kf):
        fold_start = datetime.now()
        print 'Fold %d' % (i+1)
        X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]

        cv_score = 0
        for j in xrange(args['n_bags']):
            bag_start = datetime.now()
            print 'Bag %d' % (j+1)
            np.random.seed(10*i + 66*j + 666)
            model = nn_model()
            fit = model.fit_generator(generator = batch_generator(X_train.values, y_train, 128, True),
                                      nb_epoch = nepochs,
                                      samples_per_epoch = X_train.shape[0],
                                      verbose = 0,
                                      validation_data = (X_val.values, y_val),
                                      callbacks=[EarlyStopping(monitor='val_loss', patience=10), 
                                                 ModelCheckpoint('NNcheckpoint/keras-regressor-fold' + str(i+1) + '-bag' + str(j+1) + '.check', monitor='val_loss', save_best_only=True, verbose=0)])

            best_fitted_model = load_model('NNcheckpoint/keras-regressor-fold' + str(i+1) + '-bag' + str(j+1) + '.check')
            oof_train[test_index] = best_fitted_model.predict_generator(batch_generatorp(X_val.values, 500, False), val_samples=X_val.shape[0])
            bag_score = log_loss(y_val, oof_train[test_index])
            print 'fold-' + str(i+1) + '-bag-' + str(j+1) + '-eval-mlogloss: %.6f' % bag_score

            oof_test_skf = oof_test_skf + best_fitted_model.predict(test_x.values)
            cv_score = cv_score + bag_score
            print '%dth bag is done' % (j+1)
            print 'Elapsed time: ', datetime.now() - bag_start
            print '\n'

        cv_sum = cv_sum + cv_score
        print 'fold-' + str(i+1) + '-eval-logloss: %.6f' % (cv_score / args['n_bags'])
        print '%dth fold is done' % (i+1)
        print 'Elapsed time: ', datetime.now() - fold_start
        print '\n'

    print 'training is done: ', datetime.now() - start

    oof_test = oof_test_skf / (NFOLDS * args['n_bags'])
    score = cv_sum / (NFOLDS * args['n_bags'])
    print 'Average eval-mlogss: %.6f' % score

    print "Writing oof_test meta feature"
    events = [30018, 30021, 30024, 30027, 30039, 30042, 30045, 30048, 36003, 45003]
    header = ['event_' + str(i) for i in events]
    submission = pd.DataFrame(oof_test, columns=header)
    dtest = pd.read_csv('data/test_ids.csv', header=None, names=['id'])
    submission = pd.concat([dtest['id'], submission], axis=1)
    sub_file = 'nn_10fold_'+str(args['n_bags'])+'bag_'+str(args['feature_type'])+'_l1_'+str(layer1)+'_l2_'+str(layer2)
    submission.to_csv('results/' + sub_file + '_oof_test.csv', index=False)

    print 'Writing oof_train meta feature'
    pd.DataFrame(oof_train).to_csv('results/' + sub_file + '_oof_train.csv', index=False, header=False)


