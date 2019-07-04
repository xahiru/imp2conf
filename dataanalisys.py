import time
from surprise.dump import dump
from surprise.dump import load
import pandas as pd
import matplotlib.pyplot as plt
import os
# from impt import ImpliciTrust

def laod_saved_preds(file_path='ml-100k'):
	predictions, algo = load('results/SVDpp_ml-100k_n_factors_20_n_epochs_20compare3')
	return predictions, algo

def load_dataset_df(file_path='ml-100k'):
	file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
	return pd.read_csv(filename, sep='\t', names=['itemID', 'userID', 'rating', 'timestamp'], header=0)

predictions, algo = laod_saved_preds()

for p in predictions[:10]:
		# print(t.ur[t.to_inner_uid(p.uid)])
		print(p)

print(algo.single_item_trustlist(249))
print(algo.single_user_trustlist(543))