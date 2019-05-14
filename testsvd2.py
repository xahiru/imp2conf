from surprise import NMF
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Trainset
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt 
import copy as cp
from joblib import Parallel
from joblib import delayed
import multiprocessing
import time

# Load the movielens-100k dataset (download it if needed),
# data = Dataset.load_builtin('jester')
dataset_name ='ml-100k'
data = Dataset.load_builtin('ml-100k')
# data = Dataset.load_builtin('ml-latest-small')
# data = Dataset.load_builtin('ml-1m')
myalgo = False
n_factors = 20
i_imp_factors = False
# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data,random_state=100, test_size=.25)

#loggin param details
print(dataset_name)
print('myalgo =' +str(myalgo))
print('n_factors =' +str(n_factors))
# print('i_imp_factors =' +str(i_imp_factors))


# We'll use the famous SVD algorithm.
algo = SVDpp(n_factors= n_factors, i_imp_factors=True, random_state =100)
print('i_imp_factors =' +str(i_imp_factors))
start = time.time()
if myalgo:
	print(myalgo)
	algo2 = SVDpp(n_factors= n_factors, n_epochs=20,random_state =100)

	raw2inner_id_users = {}
	raw2inner_id_items = {}
	ur = defaultdict(list)
	ir = defaultdict(list)
	ircount = {}
	n_jobs = multiprocessing.cpu_count()
	print('n_jobs ='+str(n_jobs))

	

	def set_ur(ii,rr,trainset, num_items_rated_by_user):
		item_len = len(trainset.ir[ii])
		item_pop = item_len/trainset.n_items
		first_potion = rr/num_items_rated_by_user
		return (ii, 1- (first_potion*item_pop))

	def get_ur(x,ur, trainset,n_jobs):
		num_items_rated_by_user = len(trainset.ur[x])
		delayed_list = (delayed(set_ur)(ii, rr, trainset, n_jobs)
	                        for ii, rr in trainset.ur[x])
		out = Parallel(n_jobs=n_jobs, pre_dispatch='2*n_jobs')(delayed_list)
		ur[x].append(out)

	delayed_list = (delayed(get_ur)(x, ur, trainset,n_jobs)
	                        for (x) in range(trainset.n_users))
	Parallel(n_jobs=n_jobs, pre_dispatch='2*n_jobs')(delayed_list)
	print('ur')
	print(ur)

	# for x in range(trainset.n_users):
	# 	num_items_rated_by_user = len(trainset.ur[x])
	# 	for ii,rr in trainset.ur[x]:
	# 		item_len = len(trainset.ir[ii])
	# 		item_pop = item_len/trainset.n_items
	# 		first_potion = rr/num_items_rated_by_user
	# 		ur[x].append((ii, 1-(first_potion* item_pop)))


	for x in range(trainset.n_items):
		num_users_rated_the_item = len(trainset.ir[x])
		for ii,rr in trainset.ir[x]:
			item_len = len(trainset.ur[ii])
			u_pop = item_len/trainset.n_users
			first_potion = rr/num_users_rated_the_item
			ir[x].append((ii, 1-(first_potion* u_pop)))

	for u,i,r in trainset.all_ratings():
		raw2inner_id_users[trainset.to_raw_uid(u)] = u
		raw2inner_id_items[trainset.to_raw_iid(i)] = i

	t = Trainset(ur,
			ir,
			trainset.n_users,
			trainset.n_items,
			trainset.n_ratings,
			trainset.rating_scale,
			trainset.offset,
			raw2inner_id_users,
			raw2inner_id_items)


# for uu,ii,rr in t.all_ratings():
# 	print(" ratings {}, {}, {}".format(uu,ii,rr))

# print(t.n_users)
# print(t.n_items)
# print(t.n_ratings)
	algo2.fit(t)
	print('algo2.pu')
	print(algo2.pu.shape)
	print('algo2.qi')
	print(algo2.qi.shape)

algo.fit(trainset)
print('>>>>>>>before')
print('algo.pu')
print(algo.pu.shape)
print('algo.qi')
print(algo.qi.shape)

if myalgo:
	print('>>>>>>>algo.pu - algo2.pu')
	algo.pu = algo.pu - algo2.pu
	algo.qi = algo.qi - algo2.qi
	print('>>>>>>>after')
	print(algo.pu)
	print(algo.qi)

predictions = algo.test(testset)

# for p in predictions:
# 	# print(t.ur[t.to_inner_uid(p.uid)])
# 	print(p)

# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)

print('time.time() - start')
print(time.time() - start)
'''
# plt.bar(ircount{:},100)


# plt.hist(list())
# plt.hist(list(ircount.values()))
# plt.show()
'''