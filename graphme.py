import numpy as np
import matplotlib.pyplot as plt 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from os.path import join
from surprise.builtin_datasets import get_dataset_dir
import pandas as pd


dataset_name ='ml-100k'

def load_dataset_for_stats(dataset_name='ml-100k'):
	if dataset_name == 'ml-100k':
		# file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
		path=join(get_dataset_dir(), 'ml-100k/ml-100k/u.data')
		ratings_df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
	elif dataset_name == 'ml-1m':
		path =join(get_dataset_dir(), 'ml-1m/ml-1m/ratings.dat')
		# file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
		ratings_df = pd.read_table(path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'])
	elif dataset_name == 'ml-latest-small':
		path=join(get_dataset_dir(), 'ml-latest-small/ratings.csv')
		ratings_df = pd.read_table(path, sep=',', names=['user_id', 'item_id', 'rating', 'timestamp'])
	elif dataset_name == 'jester':
		path=join(get_dataset_dir(), 'jester/jester_ratings.dat')
		ratings_df = pd.read_table(path, sep='\t\t', names=['user_id', 'item_id', 'rating'])
	else:
		print('No dataset name given')
	return ratings_df

def graph_user_average_ratings_ply(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	table = FF.create_table(ratingdf)
	py.plot(table, filename='rating-data-sample')
	# print(ratingdf.head(100))
# graph_user_average_ratings_ply(dataset_name)

def graph_user_average_ratings_matplot(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	plt.hist(ratingdf[0:1000], bins=np.arange(ratingdf[0:10].rating.min(), ratingdf[0:10].rating.max()+1))
	# plt.hist(ratingdf[0:100], bins=len(ratingdf[0:100].user_id))
	plt.show()


graph_user_average_ratings_matplot(dataset_name)