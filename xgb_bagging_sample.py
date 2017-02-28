#!/usr/bin/env python -W ignore::DeprecationWarning

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold


def xgb_fit_predict(args):
    start = datetime.now()
    print('\nStart reading data')
    print start

    feature_type = args['feature_type']
    train_x = pd.read_csv('data/' + feature_type + '_dtrain.csv', header=None)
    test_x = pd.read_csv('data/' + feature_type + '_dtest.csv', header=None)
    train_y = pd.read_csv('data/ytrain.txt', header=None).values[:, 0]

    ntrain = train_x.shape[0]
    ntest = test_x.shape[0]

    print 'Dim of the training set: ', train_x.shape
    print 'Dim of the test set: ', test_x.shape

    num_class = 10
    NFOLDS = 10
    cv_sum = 0

    oof_train = np.zeros((ntrain, num_class))
    oof_test = np.zeros((ntest, num_class))
    oof_test_skf = np.zeros((ntest, num_class))

    d_train_full = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x)

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
            params = {
                'seed': (i*10 + j*33 + 6),
                'colsample_bytree': args['colsample_bytree'],
                'silent': 1,
                'subsample': args['subsample'],
                'learning_rate': args['eta'],
                'max_depth': args['max_depth'],
                'min_child_weight': args['min_child_weight'],
                'gamma': args['gamma'],
                'objective': 'multi:softprob',
                'num_class': 10,
                'eval_metric': 'mlogloss',
                'booster': 'gbtree'}

            d_train = xgb.DMatrix(X_train, label=y_train)
            d_valid = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(d_train, 'train'), (d_valid, 'eval')]

            clf = xgb.train(params,
                            d_train,
                            100000,
                            watchlist,
                            early_stopping_rounds=60,
                            verbose_eval=30)

            oof_train[test_index] = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)

            bag_score = log_loss(y_val, oof_train[test_index])
            print 'fold-' + str(i+1) + '-bag-' + str(j+1) + '-eval-mlogloss: %.6f' % bag_score

            oof_test_skf = oof_test_skf + clf.predict(d_test, ntree_limit=clf.best_ntree_limit)
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
    sub_file = 'xgb_10fold_'+str(args['n_bags'])+'bag_'+str(args['feature_type'])+'_eta_'+str(args['eta'])+'_max_depth_'+str(args['max_depth'])
    submission.to_csv('results/' + sub_file + '_oof_test.csv', index=False)

    print 'Writing oof_train meta feature'
    pd.DataFrame(oof_train).to_csv('results/' + sub_file + '_oof_train.csv', index=False, header=False)

