from surprise import AlgoBase
from surprise import SVD
from surprise import SVDpp
from surprise import SVDppimp
from surprise import accuracy
from surprise import Dataset
from surprise.model_selection import train_test_split
from collections import defaultdict
import copy as cp
from surprise.dump import dump
import pandas as pd
import numpy as np


class ImpliciTrust(AlgoBase):

    def __init__(self, trainset, testset, algo, final_algo, compare=None, binary=True, parellel=False, random_state=100):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.trainset = trainset
        self.testset = testset
        self.final_algo = final_algo
        self.final_algo_testset = cp.deepcopy(testset)
        self.final_algo_trainset = cp.deepcopy(trainset)
        self.orginal_pu = []
        self.orginal_qi = []
        self.algo = algo
        self.binary = binary
        self.parellel = parellel
        self.random_state = random_state
        self.pu = []
        self.qu = []
        self.t_all_rating = []
        self.compare = compare if compare is not None else 0
 
    def fit(self, trainset):
        print('new trust')
        self.get_all_rating_trust(self.trainset)
        print('fitting')
        self.final_algo.fit(self.final_algo_trainset)
        print('self.trainset.ur[543]')
        print(self.trainset.ur[543])
        print('self.trainset.ir[606]')
        print(self.trainset.ir[249])
        print('copying')
        self.orginal_pu = cp.deepcopy(self.final_algo.pu)
        self.orginal_qi = cp.deepcopy(self.final_algo.qi)
        print('setting ur n ir')
        self.set_ur(trainset, self.binary)
        self.set_ir(trainset, self.binary)
        print('self.trainset.ur[543]')
        print('inner id of .ur[543]')
        print(trainset.to_raw_uid(543))
        print(self.trainset.ur[543])
        for x,_ in trainset.ur[543]:
            print(trainset.to_raw_iid(x))
        print('final fit')
        self.algo.fit(self.trainset)
        self.pu = self.algo.pu
        self.qi = self.algo.qi
        print('done fitting with modified trainset')

    def set_ur(self, trainset, binary=True):
        ir = defaultdict(list)
        if binary:
            for item_x, x_ir_list in trainset.ir.items():
                item_list = []
                item_pop = len(x_ir_list)/trainset.n_items
                for user, rating in x_ir_list:
                    u_pop = 1/len(trainset.ur[user])
                    item_list.append((user,u_pop * item_pop))
                    # print((user,u_pop * item_pop))
                ir[item_x] = item_list
            self.trainset.ir = ir
        else:
            for item_x in range(trainset.n_items):
                item_list = []
                item_pop = len(trainset.ir[item_x])/trainset.n_items
                for user, rating in trainset.ir[item_x]:
                    u_pop = rating/len(trainset.ur[user])
                    item_list.append((user,u_pop * item_pop))
                    # print((user,u_pop * item_pop))
                ir[item_x] = item_list
            self.trainset.ir = ir
    def set_ir(self, trainset, binary=True):
        ur = defaultdict(list)
        if binary:
            for user_x, x_ur_list in trainset.ur.items():
                user_list = []
                user_pop = 1/len(x_ur_list)
                for item, rating in x_ur_list:
                    item_pop = len(trainset.ir[item])/trainset.n_items
                    user_list.append((item, item_pop * user_pop))
                    # print((item, item_pop * user_pop))
                ur[user_x] = user_list
            self.trainset.ur = ur
            print('done set_ir')
        else:
            for user_x in range(trainset.n_users):
                user_list = []
                user_pop_r_len = len(trainset.ur[user_x])
                for item, rating in trainset.ur[user_x]:
                    user_pop = rating/user_pop_r_len
                    item_pop = len(trainset.ir[item])/trainset.n_items
                    user_list.append((item, item_pop * user_pop))
                    # print((item, item_pop * user_pop))
                ur[user_x] = user_list
            self.trainset.ur = ur
            print('done set_ir')

    def get_all_rating_trust(self, trainset, binary=True):
        df = pd.DataFrame(trainset.all_ratings(), columns=['user_id', 'item_id', 'rating'])
        print(df.head())
        # rx = df.user_id.unique().tolist()
        # ry = df.user_id.value_counts()
        # print('ry')
        # print(ry)
        # print('>>>>>>rx')
        # print(rx)
        # u_mean = np.sum(ry)/len(ry)
        # df['u_mean'] = u_mean
        
        # df['u_pop'] = ry[df.user_id]
        df['u_count'] = df['user_id'].map(df['user_id'].value_counts())
        df['i_count'] = df['item_id'].map(df['item_id'].value_counts())
        df['u_pop'] = 1/df.u_count
        df['nbu_pop'] = df.rating/df.u_count
        df['item_pop'] = df.i_count/trainset.n_items
        df['bitem_pop'] = 1/trainset.n_items
        df['nbitem_pop'] = 1/df.i_count
        df['nbitem_popc'] = df.rating/df.i_count
        df['trust'] = df.u_pop * df.item_pop
        print(df)
        self.t_all_rating_df = df


    def single_user_trustlist(self, u, i):
        print('retreving u trust')
        print(str(u))
        print(str(i))
        for it,t in self.trainset.ur[u]:
            # print('it')
            # print(str(it))
            # print(str(t))
            if it==i:
                return t
        return 0

    def single_item_trustlist(self, i):
    	return self.trainset.ir[i]
    
    #though this is mendatory it is nevergonna be used except for testing
    def estimate(self, u, i):
    	if self.compare == 0:
    		return self.algo.estimate(u,i)
    	if self.compare == 1:
    		return self.final_algo.estimate(u,i)
    	if self.compare == 2:
    		self.final_algo.pu = self.orginal_pu * self.pu
    		self.final_algo.qi = self.orginal_qi * self.qi
    		return self.final_algo.estimate(u,i)
    	if self.compare == 3:
    		self.final_algo.pu = self.orginal_pu - self.pu
    		self.final_algo.qi = self.orginal_qi - self.qi
    		return self.final_algo.estimate(u,i)



dataset_name ='ml-100k'
# dataset_name ='jester'
# dataset_name ='ml-1m'
# dataset_name ='ml-latest-small'
compare = 3
n_factors = 20
n_epochs = 20
save = False
binary=True
data = Dataset.load_builtin(dataset_name)
trainset, testset = train_test_split(data,random_state=100, test_size=.20)
algo_name = 'SVD_'
algo = SVD(random_state=100)
final_algo = SVD(random_state=100)
print('start')
myalgo = ImpliciTrust(trainset, testset, algo, final_algo, compare=compare, binary=binary)
myalgo.fit(trainset)
print(dataset_name)
print('compare ='+str(compare))
myalgo.single_user_trustlist(543, 249)
# myalgo.single_item_trustlist(249)
print('myalgo.pu')
print(myalgo.pu)
print('myalgo.qi')
print(myalgo.qi)
print('binary')
print(str(binary))


predictions = myalgo.test(testset)
for p in predictions[:10]:
	# print(p)
    print(p)
    if int(p[0])== '543':
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>543 found')
        print('543 found')
    print(myalgo.single_user_trustlist(int(p[0]), int(p[1])))
    # print(myalgo.single_item_trustlist( int(p[1])))
accuracy.rmse(predictions)
accuracy.mae(predictions)

df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'ratings_ui', 'estimated_Ratings', 'details'])
def get_Iu(uid):
    """Return the number of items rated by given user"""
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """Return the number of users that have rated given item"""
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:  # item was not part of the trainset
        return 0 
df['Ui'] = df.user_id.apply(get_Iu)
df['Iu'] = df.item_id.apply(get_Ui)
df['ERROR IN PREDICTION'] = abs(df.estimated_Ratings - df.ratings_ui)

print(df.head())
best_predictions = df.sort_values(by='ERROR IN PREDICTION')[:10]
print('best_predictions')
print(best_predictions)
worst_predictions = df.sort_values(by='ERROR IN PREDICTION')[-10:]
print('worst_predictions')
print(worst_predictions)
print(dataset_name)
print('compare ='+str(compare))
print('binary')
print(str(binary))
print(df.loc[df['user_id'] == '812'])

if save:
	dump('results/'+algo_name+dataset_name+'_n_factors_'+str(n_factors)+'_n_epochs_'+str(n_epochs)+'compare'+str(compare), predictions=predictions, algo=myalgo, verbose=1)

