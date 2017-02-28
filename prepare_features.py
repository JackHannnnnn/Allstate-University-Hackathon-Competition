import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD


train_file = 'train.csv'
test_ids_file = 'test.csv'

train = pd.read_csv(train_file)
test_ids = pd.read_csv(test_ids_file)
print 'Dim of raw train: ',  train.shape
print 'Dim of raw test ids: ', test_ids.shape

train_data = train[~train['id'].isin(test_ids['id'])]
test_data = train[train['id'].isin(test_ids['id'])]
del train
print 'Num of train rows in raw training data: ', train_data.shape[0]
print 'Num of test rows in raw training data: ', test_data.shape[0]

train_data = train_data.sort_values(['id', 'timestamp'])
test_data = test_data.sort_values(['id', 'timestamp'])

def construct_features(data, make_target=True):
    grouped_data = []
    
    for key, group in data.groupby('id'):
        if make_target == True:
            target_event = group.iloc[-1]
            group = group.iloc[:(group.shape[0]-1)]
        
        output_record = {'id': key, 
                         'num_events': group.shape[0], 
                         'num_unique_events': group['event'].unique().shape[0]}
        
        first_event = group.iloc[0]
        last_event = group.iloc[group.shape[0]-1]
        second_last_event = group.iloc[group.shape[0]-2]
        output_record['first_event'] = first_event['event']
        output_record['last_event'] = last_event['event']
        output_record['second_last_event'] = second_last_event['event']
      
        event_count = {30018: 0, 30021: 0, 30024: 0, 30027: 0, 30039: 0, 
                       30042: 0, 30045: 0, 30048: 0, 36003: 0, 45003: 0}
        for event in group['event']:
            event_count[event] = event_count[event] + 1
        for e_id, e_count in event_count.items():
            output_record['num_'+str(e_id)] = e_count
        
        if make_target == True:
            output_record['target_event'] = target_event['event']
        else:
            output_record['target_event'] = 'NA'
        grouped_data.append(output_record)

    return grouped_data


print '\nStart constructing basic features...'
start = datetime.now()
dtrain = pd.DataFrame(construct_features(train_data, make_target=True))
dtest = pd.DataFrame(construct_features(test_data, make_target=False))
del train_data, test_data
print 'Basic features construction are done!'
print 'Time elapsed: ', datetime.now() - start


# Cluster event types into three types and prepare relevant features
print 'Prepare features relevant to event types'
event_type_mapping = {30018: 'type_30', 30021: 'type_30', 30024: 'type_30', 
                      30027: 'type_30', 30039: 'type_30', 30042: 'type_30',
                      30045: 'type_30', 30048: 'type_30', 36003: 'type_36',
                      45003: 'type_45'}

dtrain['first_event_type'] = dtrain['first_event'].map(event_type_mapping)
dtrain['last_event_type'] = dtrain['last_event'].map(event_type_mapping)
dtrain['second_last_event_type'] = dtrain['second_last_event'].map(event_type_mapping)

dtest['first_event_type'] = dtest['first_event'].map(event_type_mapping)
dtest['last_event_type'] = dtest['last_event'].map(event_type_mapping)
dtest['second_last_event_type'] = dtest['second_last_event'].map(event_type_mapping)

type_30_columns = ['num_30018', 'num_30021', 'num_30024', 'num_30027', 
                   'num_30039', 'num_30042', 'num_30045', 'num_30048']
dtrain['num_type_30'] = dtrain[type_30_columns].sum(axis=1)
dtest['num_type_30'] = dtest[type_30_columns].sum(axis=1)

print 'Feature preparatoin is done!'
print 'Dim of dtrain:', dtrain.shape
print 'Dim of dtest:', dtest.shape

# Save test ids and ytrain 
test_ids.to_csv('data/test_ids.csv', index=False)
ytrain = dtrain['target_event']
ytrain = pd.factorize(ytrain, sort=True)[0]
pd.Series(ytrain).to_csv('data/ytrain.txt', index=False, header=False)


# Encoding numeric features and categorical features
ntrain = dtrain.shape[0]
data = pd.concat([dtrain, dtest], axis=0, ignore_index=True)
del data['id'], data['target_event'], dtrain, dtest

def type_process(data):
    for f in data.columns:
        if not f.startswith('num'):
            data[f] = data[f].astype(np.object)
        else:
            data[f] = data[f].astype(np.int16)
type_process(data)


numeric_features = [f for f in data.columns if f.startswith('num')]
cat_features = [f for f in data.columns if not f.startswith('num')]

# Scale numeric features
scaled_numeric_feats = StandardScaler().fit_transform(data[numeric_features])

# One-hot encoding for categorical features
onehot_cat_feats = pd.get_dummies(data[cat_features]).values

output = np.hstack([data[numeric_features].values, onehot_cat_feats])
pd.DataFrame(output[:ntrain]).to_csv('data/ori_num_onehot_cat_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/ori_num_onehot_cat_dtest.csv', index=False, header=False)

output = np.hstack([scaled_numeric_feats, onehot_cat_feats])
pd.DataFrame(output[:ntrain]).to_csv('data/scaled_num_onehot_cat_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/scaled_num_onehot_cat_dtest.csv', index=False, header=False)

# Ordered encoding for categorical features
ordered_cat_feats = np.zeros([data.shape[0], len(cat_features)])
for i, cat in enumerate(cat_features):
    ordered_cat_feats[:, i] = pd.factorize(data[cat], sort=True)[0]
    
output = np.hstack([data[numeric_features].values, ordered_cat_feats])
pd.DataFrame(output[:ntrain]).to_csv('data/ori_num_ordered_cat_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/ori_num_ordered_cat_dtest.csv', index=False, header=False)

output = np.hstack([scaled_numeric_feats, ordered_cat_feats])
pd.DataFrame(output[:ntrain]).to_csv('data/scaled_num_ordered_cat_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/scaled_num_ordered_cat_dtest.csv', index=False, header=False)

# SVD for scaled numeric features and categorical features
numeric_onehot_data = np.hstack([scaled_numeric_feats, onehot_cat_feats])
n_components = 32
svd = TruncatedSVD(n_components)
output = svd.fit_transform(numeric_onehot_data)
print "Explained variance ratio: %.5f" % np.sum(svd.explained_variance_ratio_)

pd.DataFrame(output[:ntrain]).to_csv('data/svd_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/svd_dtest.csv', index=False, header=False)

# Counts encoding for categorical features
counts_cat_feats = np.zeros([data.shape[0], len(cat_features)])
for i, cat in enumerate(cat_features):
    counts = data[cat].value_counts()
    counts_cat_feats[:, i] = data[cat].map(counts)
counts_cat_feats = scale(counts_cat_feats)

output = np.hstack([data[numeric_features].values, counts_cat_feats])
pd.DataFrame(output[:ntrain]).to_csv('data/ori_num_counts_cat_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/ori_num_counts_cat_dtest.csv', index=False, header=False)

output = np.hstack([scaled_numeric_feats, counts_cat_feats])
pd.DataFrame(output[:ntrain]).to_csv('data/scaled_num_counts_cat_dtrain.csv', index=False, header=False)
pd.DataFrame(output[ntrain:]).to_csv('data/scaled_num_counts_cat_dtest.csv', index=False, header=False)

print 'All data sets are generated!'