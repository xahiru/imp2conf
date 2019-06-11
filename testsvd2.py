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
import surprise.dump

# Load the movielens-100k dataset (download it if needed),
# data = Dataset.load_builtin('jester')
dataset_name ='ml-100k'
# dataset_name ='ml-1m'
# dataset_name ='ml-latest-small'
# dataset_name ='jester'
data = Dataset.load_builtin(dataset_name)
# data = Dataset.load_builtin('ml-latest-small')
# data = Dataset.load_builtin('ml-1m')
myalgo = True
n_factors = 20
i_imp_factors = False
# sample random trainset and testset
# test set is made of 20% of the ratings.
trainset, testset = train_test_split(data,random_state=100, test_size=.20)

#loggin param details
print(dataset_name)
print('myalgo =' +str(myalgo))
print('n_factors =' +str(n_factors))
# print('i_imp_factors =' +str(i_imp_factors))
print('first_potion = 1*item_pop')


# We'll use the SVD ++ algorithm.
algo = SVDpp(n_factors= n_factors, i_imp_factors=i_imp_factors, random_state =100)
print('i_imp_factors =' +str(i_imp_factors))
start = time.time()
if myalgo:
	trainset2 = cp.deepcopy(trainset)
	algo2 = SVDpp(n_factors= n_factors, n_epochs=20,random_state =100)

	raw2inner_id_users = {}
	raw2inner_id_items = {}
	ur = defaultdict(list)
	ir = defaultdict(list)
	# print('trainset2.ur')
	# print(trainset2.ur)
	# print('trainset2.ir')
	# print(trainset2.ir)
	print('trainset2.n_ratings3 = '+ str(trainset2.n_ratings))
	# ircount = {}
	# n_jobs = multiprocessing.cpu_count()
	# print('n_jobs ='+str(n_jobs))

	for item_x in range(trainset2.n_items):
		item_list = []
		item_pop = len(trainset2.ir[item_x])/trainset2.n_items
		for user, rating in trainset2.ir[item_x]:
			u_pop = 1/len(trainset2.ur[user])
			# print(item)
			# print(rating)
			item_list.append((user,u_pop * item_pop))
		ir[item_x] = item_list

	# assert ir == trainset2.ir
	trainset2.ir = ir

	for user_x in range(trainset2.n_users):
		user_list = []
		u_pop = 1/len(trainset2.ur[user_x])
		for item, rating in trainset2.ur[user_x]:
			item_pop = len(trainset2.ir[item])/trainset2.n_items
			# print(item)
			# print(rating)
			user_list.append((item,1))
		ur[user_x] = user_list

	# assert ur == trainset2.ur
	trainset2.ur = ur

	# for user_x in range(trainset2.n_users):
	# 	item_len = len(trainset2.ur[x])
	# 	item_pop = item_len/trainset2.n_items
	# 	for uu,rr in trainset2.ir[x]:
	# 		num_items_rated_by_user = len(trainset2.ur[uu])
	# 		first_potion = 1/num_items_rated_by_user
	# 		# ir[x].append((uu, 1-(first_potion* u_pop)))
	# 		ir[x].append((uu, (first_potion* item_pop)))


	# def set_ur(ii,rr,trainset, num_items_rated_by_user):
	# 	item_len = len(trainset2.ir[ii])
	# 	item_pop = item_len/trainset2.n_items
	# 	# first_potion = 1/num_items_rated_by_user
	# 	first_potion = 1
	# 	return (ii, (first_potion*item_pop))

	# def get_ur(x,ur, trainset,n_jobs):
	# 	num_items_rated_by_user = len(trainset2.ur[x])
	# 	delayed_list = (delayed(set_ur)(ii, rr, trainset2, n_jobs)
	#                         for ii, rr in trainset2.ur[x])
	# 	out = Parallel(n_jobs=n_jobs, pre_dispatch='2*n_jobs')(delayed_list)
	# 	ur[x].append(out)

	# delayed_list = (delayed(get_ur)(x, ur, trainset2,n_jobs)
	#                         for (x) in range(trainset2.n_users))
	# Parallel(n_jobs=n_jobs, pre_dispatch='2*n_jobs')(delayed_list)
	# print('ur')
	# print(ur)


	# for x in range(trainset2.n_items):
	# 	# num_users_rated_the_item = len(trainset.ur[x])
	# 	item_len = len(trainset2.ur[x])
	# 	item_pop = item_len/trainset2.n_items
	# 	for uu,rr in trainset2.ir[x]:
	# 		num_items_rated_by_user = len(trainset2.ur[uu])
	# 		first_potion = 1/num_items_rated_by_user
	# 		# ir[x].append((uu, 1-(first_potion* u_pop)))
	# 		ir[x].append((uu, (first_potion* item_pop)))

	# for u,i,r in trainset2.all_ratings():
	# 	raw2inner_id_users[trainset2.to_raw_uid(u)] = u
	# 	raw2inner_id_items[trainset2.to_raw_iid(i)] = i

	# t = Trainset(ur,
	# 		ir,
	# 		# trainset2.n_users,
	# 		943,
	# 		# trainset2.n_items,
	# 		1656,
	# 		trainset2.n_ratings,
	# 		trainset2.rating_scale,
	# 		trainset2.offset,
	# 		raw2inner_id_users,
	# 		raw2inner_id_items)
	# assert trainset.n_users == t.n_users
	# assert trainset.n_items == t.n_items
	# assert trainset.ur == t.ur

# for uu,ii,rr in t.all_ratings():
# 	print(" ratings {}, {}, {}".format(uu,ii,rr))

# print(t.n_users)
# print(t.n_items)
# print(t.n_ratings)
	algo2.fit(trainset2)
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>algo2.fit(trainset2)<<<<<<<before')
	print('algo2.pu')
	print(algo2.pu)
	print('algo2.qi')
	print(algo2.qi)

algo.fit(trainset)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>algo.fit(trainset)<<<<<<<before')
print('algo.pu')
print(algo.pu)
print('algo.qi')
print(algo.qi)

if myalgo:
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>algo.pu - algo2.pu<<<<<after<<<<<<')
	algo.pu = algo.pu - algo2.pu
	algo.qi = algo.qi - algo2.qi
	print('algo.pu')
	print(algo.pu)
	print('algo.qi')
	print(algo.qi)
	print('>>>>>>>end after<<<<<<<<<<<<<<<<<<<<<')

predictions = algo.test(testset)

# for p in predictions:
# 	# print(t.ur[t.to_inner_uid(p.uid)])
# 	print(p)

# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)
accuracy.fcp(predictions)

print('time.time() - start')
print(time.time() - start)
'''
# plt.bar(ircount{:},100)


# plt.hist(list())
# plt.hist(list(ircount.values()))
# plt.show()
'''
# dump('myalgosvd', predictions=predictions, algo=algo, verbose=1)
# load()