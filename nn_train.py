import nn_bagging_sample

nn_params = [
    {'feature_type': 'scaled_num_onehot_cat',
     'n_bags': 2,
     'layer1': 97,
     'layer2': 30,
     'layer1_dp': 0.15,
     'layer2_dp': 0.1},
    
    {'feature_type': 'scaled_num_onehot_cat',
     'n_bags': 2,
     'layer1': 51,
     'layer2': 39,
     'layer1_dp': 0.1,
     'layer2_dp': 0.1},
    
    {'feature_type': 'scaled_num_counts_cat',
     'n_bags': 2,
     'layer1': 164,
     'layer2': 112,
     'layer3': 64,
     'layer1_dp': 0.15,
     'layer2_dp': 0.15,
     'layer3_dp': 0.1},
    
    {'feature_type': 'svd',
     'n_bags': 2,
     'layer1': 169,
     'layer2': 117,
     'layer1_dp': 0.3241,
     'layer2_dp': 0.3369}
]

for params in nn_params:
    print 'NN params:'
    print params
    nn_bagging_sample.nn_fit_predict(params)
    print 'One nn model is trained!'
print 'All nn models are done!'