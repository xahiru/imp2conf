import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab

# import plotly.offline as plot
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
def graph_by(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	data = [go.Histogram(x = ratingdf.user_id)]
	layout = dict(xaxis = dict(title = 'user_id'), yaxis = dict(title ='Frequency'))        
	fig = dict(data = data, layout = layout)                      
	py.plot(fig, 'Distribution of users')
	
def graph_plt(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	x = ratingdf.user_id
	num_bins = len(ratingdf.user_id.unique())
	n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
	plt.show()

def graph_average_plt(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	x = ratingdf.user_id
	num_bins = len(ratingdf.user_id.unique())
	n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
	plt.show()

def graph_distr_pl(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	# example data
	mu = ratingdf.rating.mean() # mean of distribution
	sigma = ratingdf.rating.std() # standard deviation of distribution
	x = ratingdf.rating
	num_bins = len(ratingdf.user_id.unique())
	# the histogram of the data
	n, bins, patches = plt.hist(x, num_bins, density=1, facecolor='blue', alpha=0.5)

	# add a 'best fit' line
	y = mlab.normpdf(bins, mu, sigma)
	plt.plot(bins, y, 'r--')
	plt.xlabel('rating scale')
	plt.ylabel('Probability')
	plt.title(r'Histogram of IQ: $\mu=mu$, $\sigma=sigma$')

	# Tweak spacing to prevent clipping of ylabel
	plt.subplots_adjust(left=0.15)
	plt.show()
	
def graph_bar(dataset_name):
	ratingdf = load_dataset_for_stats(dataset_name)
	# print(ratingdf.head(10))
	objects = ratingdf.user_id.unique().tolist()
	# print(objects)
	y_pos = np.arange(len(objects))
	# print(y_pos)
	# print(ratingdf.user_id.value_counts())
	# print(ratingdf.iloc[objects].rating)
	# print(ratingdf.groupby('user_id').ratings)
	performance = []

	for x in objects:
		singleuser = ratingdf[ratingdf.user_id == x]
		performance.append(singleuser.rating.mean())
		# ratingdf['user_mean'] = singleuser

	# performance = ratingdf.user_id.value_counts()
	print(performance)
	# print(ratingdf.head(100))

	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('counts')
	plt.title('User rating Frequency')
	plt.show()

def graph_by_py(dataset_name,itype='user_id'):
	ratingdf = load_dataset_for_stats(dataset_name)
	if itype == 'user_id':
		rx = ratingdf.user_id.unique().tolist()
		ry = ratingdf.user_id.value_counts()
	elif itype == 'item_id':
		rx = ratingdf.item_id.unique().tolist()
		ry = ratingdf.item_id.value_counts()
	data = [go.Bar(
            x = rx,
            y = ry
    )]
	print('average Frequency')
	print(str(np.sum(ry)/len(ry)))
	py.plot(data, filename='basic-bar')
	pass
# graph_user_average_ratings_matplot(dataset_name)
# graph_by(dataset_name)
# graph_plt(dataset_name)
# graph_distr_pl(dataset_name)
# graph_bar(dataset_name)
graph_by_py(dataset_name)
