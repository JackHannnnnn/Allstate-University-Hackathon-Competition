import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss
from scipy.optimize import minimize

print 'Start reading prediction data from level-1 models...'

ytrain = pd.read_csv('data/ytrain.txt', header=None).values[:, 0]
ytrain = pd.get_dummies(ytrain).values

nn_scaled_onehot_train1 = pd.read_csv('results/nn_10fold_2bag_scaled_num_onehot_cat_l1_51_l2_39_oof_train.csv', header=None)
nn_scaled_onehot_test1 = pd.read_csv('results/nn_10fold_2bag_scaled_num_onehot_cat_l1_51_l2_39_oof_test.csv')
nn_scaled_onehot_train2 = pd.read_csv('results/nn_10fold_2bag_scaled_num_onehot_cat_l1_97_l2_30_oof_train.csv', header=None)
nn_scaled_onehot_test2 = pd.read_csv('results/nn_10fold_2bag_scaled_num_onehot_cat_l1_97_l2_30_oof_test.csv')

nn_scaled_counts_train1 = pd.read_csv('results/nn_10fold_2bag_scaled_num_counts_cat_l1_164_l2_112_oof_train.csv', header=None)
nn_scaled_counts_test1 = pd.read_csv('results/nn_10fold_2bag_scaled_num_counts_cat_l1_164_l2_112_oof_test.csv')

nn_svd_train1 = pd.read_csv('results/nn_10fold_2bag_svd_l1_169_l2_117_oof_train.csv', header=None)
nn_svd_test1 = pd.read_csv('results/nn_10fold_2bag_svd_l1_169_l2_117_oof_test.csv')

nn = [nn_scaled_onehot_train1, 
      nn_scaled_onehot_train2, 
      nn_scaled_counts_train1,
      nn_svd_train1]

xgb_ori_counts_train1 = pd.read_csv('results/xgb_10fold_2bag_ori_num_counts_cat_eta_0.03_max_depth_8_oof_train.csv', header=None)
xgb_ori_counts_test1 = pd.read_csv('results/xgb_10fold_2bag_ori_num_counts_cat_eta_0.03_max_depth_8_oof_test.csv')
xgb_ori_counts_train2 = pd.read_csv('results/xgb_10fold_2bag_ori_num_counts_cat_eta_0.03_max_depth_9_oof_train.csv', header=None)
xgb_ori_counts_test2 = pd.read_csv('results/xgb_10fold_2bag_ori_num_counts_cat_eta_0.03_max_depth_9_oof_test.csv')

xgb_ori_onehot_train1 = pd.read_csv('results/xgb_10fold_2bag_ori_num_onehot_cat_eta_0.03_max_depth_4_oof_train.csv', header=None)
xgb_ori_onehot_test1 = pd.read_csv('results/xgb_10fold_2bag_ori_num_onehot_cat_eta_0.03_max_depth_4_oof_test.csv')

xgb_scaled_onehot_train1 = pd.read_csv('results/xgb_10fold_2bag_scaled_num_onehot_cat_eta_0.03_max_depth_6_oof_train.csv', header=None)
xgb_scaled_onehot_test1 = pd.read_csv('results/xgb_10fold_2bag_scaled_num_onehot_cat_eta_0.03_max_depth_6_oof_test.csv')

xgb_scaled_counts_train1 = pd.read_csv('results/xgb_10fold_2bag_scaled_num_counts_cat_eta_0.03_max_depth_6_oof_train.csv', header=None)
xgb_scaled_counts_test1 = pd.read_csv('results/xgb_10fold_2bag_scaled_num_counts_cat_eta_0.03_max_depth_6_oof_test.csv')

xgb_svd_train1 = pd.read_csv('results/xgb_10fold_2bag_svd_eta_0.03_max_depth_4_oof_train.csv', header=None)
xgb_svd_test1 = pd.read_csv('results/xgb_10fold_2bag_svd_eta_0.03_max_depth_4_oof_test.csv')

xgb_scaled_ordered_train1 = pd.read_csv('results/xgb_10fold_2bag_scaled_num_ordered_cat_eta_0.05_max_depth_8_oof_train.csv', header=None)
xgb_scaled_ordered_test1 = pd.read_csv('results/xgb_10fold_2bag_scaled_num_ordered_cat_eta_0.05_max_depth_8_oof_test.csv')


xg = [xgb_ori_counts_train1, 
      xgb_ori_counts_train2, 
      xgb_ori_onehot_train1, 
      xgb_scaled_onehot_train1,
      xgb_scaled_counts_train1, 
      xgb_svd_train1,
      xgb_scaled_ordered_train1]

test_data = [nn_scaled_onehot_test1, 
             nn_scaled_onehot_test2,
             nn_scaled_counts_test1,
             nn_svd_test1,
             xgb_ori_counts_test1, 
             xgb_ori_counts_test2, 
             xgb_ori_onehot_test1, 
             xgb_scaled_onehot_test1,
             xgb_scaled_counts_test1, 
             xgb_svd_test1,
             xgb_scaled_ordered_test1]


train_data = nn + xg


# Weighted average optimization
def mlog_loss(weights):
    final_pred = 0
    for i, weight in enumerate(weights):
        final_pred += weight * train_data[i].values
    return log_loss(ytrain, final_pred)

print '\nWeighted optimzation starts'
start = datetime.now()
scores = []
weights = []
for i in range(10):
    start = datetime.now()
    print i+1, 'th time '
    print start
    bounds = [(0,1 )] * len(train_data)
    starting_values = np.random.uniform(size=len(train_data))
    try:
        res = minimize(mlog_loss,
                       starting_values,
                       method='L-BFGS-B',
                       bounds=bounds)
        
        best_score = res['fun']
        best_weights = res['x']
        print 'Best score: ', best_score
        print 'Best weights: ', best_weights
        scores.append(best_score)
        weights.append(best_weights)
    except ValueError as e:
        print i+1, 'th time: ', 'Input contains inf'
    print 'Time elapsed: ', datetime.now() - start
    print '\n'

if len(scores) != 0:
	print 'Overall best score: ', scores[scores.index(np.array(scores).min())]
	print 'Overall best weights: ', weights[scores.index(np.array(scores).min())]
	print 'Time elapsed: ', datetime.now()-start

	best_w = weights[scores.index(np.array(scores).min())]

	final_test_pred = 0
	for i, w in enumerate(best_w):
		final_test_pred += w * test_data[i].iloc[:, 1:]
	
	pd.concat([test_data[0]['id'], final_test_pred], axis=1).to_csv('weighted_ave_v3.csv', index=False)
	print 'Optimization finished!'
else:
    print 'Optimization failed!'
	
print 'Final predictions are done!'