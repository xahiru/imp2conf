from surprise import SVD
import os
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import BaselineOnly
# from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
import pandas as pd
import random

# path to dataset file
file_path = os.path.expanduser('~/.surprise_data/Netflix-Dataset/training_set/training_set/')

def chooserandomnfiles(path, list_size=10, random_state=100, verbose=True):
	all_files = []
	li = []
	# random.seed(random_state)
	filesindir = os.listdir(path)
	num_files_in_dir = len(filesindir)
	for x in range(0,list_size):
		random_number = random.randint(0,num_files_in_dir)
		filename = 'mv_'+str(random_number).zfill(7)+'.txt'
		if verbose:
			print(filename)
		all_files.append(filename)
	for filename in all_files:
		filename = path+filename
		if verbose:
			print(filename)
		df = pd.read_csv(filename, sep='\t', names=['itemID', 'userID', 'rating', 'timestamp'], header=0)
		li.append(df)
	return pd.concat(li, axis=0, ignore_index=True, sort=True)
    # return 0
    # return frame = pd.concat(li, axis=0, ignore_index=True, sort=True)

		

df = chooserandomnfiles(file_path, list_size=30, random_state=100, verbose=False)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
print('df len = '+str(len(df.index)))

algo = SVD(random_state=100)
kf = KFold(n_splits=5,  random_state=100)
for trainset, testset in kf.split(data):
#     # train and test algorithm.
    algo.fit(trainset)
    print('n_users = '+str(trainset.n_users))
    print('n_items = '+str(trainset.n_items))
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    # print('hello')