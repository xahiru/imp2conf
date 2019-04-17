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


# Load the movielens-100k dataset (download it if needed),
# data = Dataset.load_builtin('jester')
data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data,random_state=100, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVDpp(random_state =100)
algo2 = SVDpp(n_epochs=20,random_state =100)
# algo2 = SVDpp(n_epochs=20,random_state =100)
# algo2 = NMF(random_state =100)

# algo = SVD(random_state =100)
# algo2 = SVD(n_epochs=20, random_state =100)


# Train the algorithm on the trainset, and predict ratings for the testset
# print()
raw2inner_id_users = {}
raw2inner_id_items = {}
ur = defaultdict(list)
ir = defaultdict(list)
ircount = {}

old_ur = cp.deepcopy(trainset.ur)
old_ir = cp.deepcopy(trainset.ir)
# print(old_ur.keys())
print(len(old_ur[2]))

for u,i,r in trainset.all_ratings():
# 	# print(" ratings {}, {}, {}".format(u,i,r))

	raw2inner_id_users[trainset.to_raw_uid(u)] = u
	raw2inner_id_items[trainset.to_raw_iid(i)] = i

# 	ur[u].append((i, r))
# 	ir[i].append((u, r))
# # print(trainset.n_users)
# # print(trainset.n_items)
# # print(trainset.n_ratings)

# for x in zip(trainset.ur):
for x in range(trainset.n_users):
	# print(trainset.ur[x])
	num_items_rated_by_user = len(trainset.ur[x])
	# sum_rating = 0
	# for i,r in trainset.ur[x]:
	# 	# print(r)
	# 	if i not in ircount:
	# 		ircount[i] = 1
	# 	else:
	# 		ircount[i] += 1
	# 	sum_rating += r
	for ii,rr in trainset.ur[x]:
		item_len = len(trainset.ir[ii])
		item_pop = item_len/trainset.n_items
		first_potion = rr/num_items_rated_by_user

		# ur[x].append((ii, rr/num_items_rated_by_user))
		ur[x].append(ii, rfirst_potion*item_pop)

for x in range(trainset.n_items):
	# print(trainset.ir[x])
	num_users_rated_the_item = len(trainset.ir[x])

	# sum_rating = 0
	# for i,r in trainset.ir[x]:
	# 	# print(r)
	# 	sum_rating += r
	for ii,rr in trainset.ir[x]:
		item_len = len(trainset.ur[ii])
		u_pop = item_len/trainset.n_users
		first_potion = rr/num_users_rated_the_item
		# ir[x].append((ii, rr/num_users_rated_the_item))
		ir[x].append(ii, first_potion* u_pop)
# print(ircount)
# print(list(ircount.values()))


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

algo.fit(trainset)
algo2.fit(t)
print('algo.pu')
print(algo.pu.shape)
print('algo.qi')
print(algo.qi.shape)

print('algo2.pu')
print(algo2.pu.shape)
print('algo2.qi')
print(algo2.qi.shape)


algo.pu = algo.pu - algo2.pu
algo.qi = algo.qi - algo2.qi

predictions = algo.test(testset)

# for p in predictions:
# 	# print(t.ur[t.to_inner_uid(p.uid)])
# 	print(p)

# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)
'''
# plt.bar(ircount{:},100)


# plt.hist(list())
# plt.hist(list(ircount.values()))
# plt.show()
'''