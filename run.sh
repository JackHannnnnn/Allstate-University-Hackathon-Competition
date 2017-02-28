#!/bin/sh


make_files() {
	echo "Make necessary folders"
	mkdir data
	mkdir results
	mkdir NNcheckpoint
}

prepare() {
  echo "\nPreparing ..."
  python prepare_features.py
}

train() {
  echo "\nTraining" $1 "models..."
  time python $1_train.py
}


echo "Start\n"

# Make necessary folders
make_files

# Prepare features
prepare

# Train xgb models
train xgb

# Train nn models
train nn

# Weighted optimization
python weighted_optimization.py

echo "Final predictions are in the file:  weighted_ave_v3.csv"