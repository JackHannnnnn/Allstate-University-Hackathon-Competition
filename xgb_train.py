from xgb_bagging_sample import xgb_fit_predict

xgb_params = [
    {'feature_type': 'ori_num_onehot_cat',
     'eta': 0.03,
     'n_bags': 2,
     'colsample_bytree': 0.5581,
     'gamma': 1.2376,
     'max_depth': 4,
     'min_child_weight': 5,
     'subsample': 0.8357},
    
    {'feature_type': 'scaled_num_onehot_cat',
     'eta': 0.03,
     'n_bags': 2,
     'colsample_bytree': 0.544,
     'gamma': 2.3173,
     'max_depth': 6,
     'min_child_weight': 3,
     'subsample': 0.628},
    
    {'feature_type': 'scaled_num_counts_cat',
     'eta': 0.03,
     'n_bags': 2,
     'colsample_bytree': 0.5917,
     'gamma': 2.4745,
     'max_depth': 6,
     'min_child_weight': 5,
     'subsample': 0.9033},
    
    {'feature_type': 'ori_num_counts_cat',
     'eta': 0.03,
     'n_bags': 2,
     'colsample_bytree': 0.5698,
     'gamma':  2.3845,
     'max_depth': 9,
     'min_child_weight': 1,
     'subsample': 0.832},
    
    {'feature_type': 'ori_num_counts_cat',
     'eta': 0.03,
     'n_bags': 2,
     'colsample_bytree': 0.4044,
     'gamma': 2.2487,
     'max_depth': 8,
     'min_child_weight': 5,
     'subsample': 0.9418},
    
    {'feature_type': 'scaled_num_ordered_cat',
     'eta': 0.05,
     'n_bags': 2,
     'colsample_bytree': 0.345,
     'gamma': 1.7278,
     'max_depth': 8,
     'min_child_weight': 2,
     'subsample': 0.7084},
    
    {'feature_type': 'svd',
     'eta': 0.03,
     'n_bags': 2,
     'colsample_bytree': 0.5216,
     'gamma': 1.9593,
     'max_depth': 4,
     'min_child_weight': 2,
     'subsample': 0.9417}
]

for params in xgb_params:
    print 'xgb params:'
    print params
    xgb_fit_predict(params)
    print '\nOne xgb model is trained!'
print '\nAll xgb models are done!'
    