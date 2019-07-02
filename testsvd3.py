from surprise import NMF
from surprise import SVD
from surprise import SVDpp
from surprise import SVDppimp
from surprise import AlgoBase
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Trainset
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import copy as cp
from joblib import Parallel
from joblib import delayed
import multiprocessing
import time
from surprise.dump import dump
import pandas as pd
from os.path import join
from surprise.builtin_datasets import get_dataset_dir
from surprise.model_selection import KFold

# Load the movielens-100k dataset (download it if needed),
dataset_name ='ml-100k'
# dataset_name ='ml-1m'
# dataset_name ='ml-latest-small'
# dataset_name ='jester'
data = Dataset.load_builtin(dataset_name)
myalgo = True
n_factors = 20
n_epochs =  20
i_imp_factors = False
pminusq = False
ptimesq = not pminusq
beta = 0.15
binary = False
# sample random trainset and testset
# test set is made of 20% of the ratings.
# trainset, testset = train_test_split(data,random_state=100, test_size=.20)


# graph_user_average_ratings_matplot(dataset_name)
#loggin param details
print(dataset_name)
print('myalgo =' +str(myalgo))
print('n_factors =' +str(n_factors))
print('n_epochs =' +str(n_epochs))
print('i_imp_factors =' +str(i_imp_factors))
# print('first_potion = 1*item_pop')
print('beta =' +str(beta))
print('pminusq =' +str(pminusq))

sum_rmse = 0
sum_mae = 0
sum_fcp = 0
kt = 0

sum_rmse_svd = 0
sum_mae_svd = 0
sum_fcp_svd = 0

kf = KFold(n_splits=5,  random_state=100)
for trainset, testset in kf.split(data):
	# We'll use the SVD ++ algorithm.
	# algo = SVDpp(n_factors= n_factors, i_imp_factors=i_imp_factors, random_state=100)
	algo = SVDppimp(n_factors= n_factors, i_imp_factors=i_imp_factors, binary=binary, random_state=100)
	trainset3 = cp.deepcopy(trainset)
	testset3 = cp.deepcopy(testset)
	algo.fit(trainset3)
	algo_original = cp.deepcopy(algo)
	algo_original = SVDppimp(n_factors= n_factors, i_imp_factors=i_imp_factors, binary=binary, random_state=100)
	algo_original.fit(trainset3)
	testset_orginal = cp.deepcopy(testset3)
	start = time.time()
	if myalgo:
		trainset2 = cp.deepcopy(trainset)
		algo2 = SVDppimp(n_factors= n_factors, n_epochs=n_epochs,random_state=100)

		raw2inner_id_users = {}
		raw2inner_id_items = {}
		ur = defaultdict(list)
		ir = defaultdict(list)
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>fold start<<<<<<<')
		print('trainset2.n_ratings = '+ str(trainset2.n_ratings))
		print('trainset2.n_items = '+ str(trainset2.n_items))
		print('trainset2.n_users = '+ str(trainset2.n_users))
		print('user:item = '+ str(trainset2.n_users/trainset2.n_items))
		# beta = trainset2.n_users/trainset2.n_items 
		# if dataset_name == 'jester':
			# beta = trainset2.n_items/trainset2.n_users
		print('beta =' +str(beta))

		for item_x in range(trainset2.n_items):
			item_list = []
			item_pop = len(trainset2.ir[item_x])/trainset2.n_items
			# item_pop = item_pop * beta
			# print('item_pop')
			# print(item_pop)
			for user, rating in trainset2.ir[item_x]:
				u_pop = 1/len(trainset2.ur[user])
				# print(item)
				# print(rating)
				item_list.append((user,u_pop * item_pop))
			ir[item_x] = item_list
		trainset2.ir = ir

		for user_x in range(trainset2.n_users):
			user_list = []
			#using 1 heare simply binarizes, meaning r=1
			u_pop = 1/len(trainset2.ur[user_x])
			for item, rating in trainset2.ur[user_x]:
				item_pop = len(trainset2.ir[item])/trainset2.n_items
				# item_pop = item_pop * beta
				# print(item)
				# print(rating)
				user_list.append((item,1))
			ur[user_x] = user_list

		trainset2.ur = ur

		algo2.fit(trainset2)


	if myalgo:
		# print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>algo.pu - algo2.pu<<<<<after<<<<<<')
		if pminusq:
			# print('algo.pu - algo2.pu')
			# print('(2-beta ) * algo.pu - beta * algo2.pu')
			# algo.pu =  (1-beta) * algo.pu - beta * algo2.pu
			# algo.qi =  (1-beta) * algo.qi - beta * algo2.qi
			algo.pu =   algo.pu - algo2.pu
			algo.qi =  algo.qi - algo2.qi
		if ptimesq:
			print('algo.pu * algo2.pu')
			algo.pu = algo.pu * algo2.pu
			algo.qi = algo.qi * algo2.qi

		print('>>>>>>>results<<<<<<<<<<<<<<<<<<<<<')

	predictions = algo.test(testset)

	print('>>>>>>>predictions<<<<<<<<<<<<<<<<<<<<<')
	for p in predictions[:10]:
		# print(t.ur[t.to_inner_uid(p.uid)])
		print(p)

	print(dataset_name)
	print('myalgo =' +str(myalgo))
	print('n_factors =' +str(n_factors))
	print('n_epochs =' +str(n_epochs))
	if pminusq:
		print('algo.pu - algo2.pu')
		# print('(2-beta ) * algo.pu - beta * algo2.pu')
		# print('(1-beta ) * algo.pu - beta * algo2.pu')
	if ptimesq:
		print('algo.pu * algo2.pu')
	# Then compute RMSE
	print('beta =' +str(beta))
	sum_rmse += accuracy.rmse(predictions, binary=binary)
	sum_mae += accuracy.mae(predictions, binary=binary)
	sum_fcp += accuracy.fcp(predictions)

	print('====SVDpp== results')
	p2 = algo_original.test(testset_orginal)
	print('>>>>>>>predictions p2<<<<<<<<<<<<<<<<<<<<<')
	for p in p2[:10]:
		print(p)
	sum_rmse_svd += accuracy.rmse(p2, binary=binary)
	sum_mae_svd += accuracy.mae(p2, binary=binary)
	sum_fcp_svd += accuracy.fcp(p2)

	print('time.time() - start')
	print(time.time() - start)
	kt += 1


print('====final averages results  myalgo==')
print('RMSE')
print(str(sum_rmse/kt))
print('mae')
print(str(sum_mae/kt))
print('sum_fcp')
print(str(sum_fcp/kt))

print('====final averages results   SVDpp==')
print('RMSE')
print(str(sum_rmse_svd/kt))
print('mae')
print(str(sum_mae_svd/kt))
print('sum_fcp')
print(str(sum_fcp_svd/kt))




# dump('results/myalgosvd'+dataset_name+'_n_factors_'+str(n_factors)+'_n_epochs_'+str(n_epochs)+'_beta_'+str(beta)+'_myalgo_'+str(myalgo)+'_pminusq_'+str(pminusq)+'_ptimesq_'+str(ptimesq), predictions=predictions, algo=algo, verbose=1)
# load()