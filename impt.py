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
        self.trainset2 = cp.deepcopy(trainset)
        self.orginal_pu = []
        self.orginal_qi = []
        self.algo = algo
        self.binary = binary
        self.parellel = parellel
        self.random_state = random_state
        self.pu = []
        self.qu = []
        # self.t_all_rating =  pd.DataFrame(trainset.all_ratings(), columns=['user_id', 'item_id', 'rating'])
        self.compare = compare if compare is not None else 0
 
    def fit(self, trainset):
        print('new trust')
        self.t_all_rating = self.get_all_rating_trust(self.trainset2)
        print('fitting')
        self.final_algo.fit(self.final_algo_trainset)
        print('copying')
        self.orginal_pu = cp.deepcopy(self.final_algo.pu)
        self.orginal_qi = cp.deepcopy(self.final_algo.qi)
        print('setting ur n ir')
        self.set_ur_new(self.trainset, self.t_all_rating)
        self.set_ir_new(self.trainset, self.t_all_rating)
        print('final fit')
        self.algo.fit(self.trainset)
        self.pu = cp.deepcopy(self.algo.pu)
        self.qi = cp.deepcopy(self.algo.qi)
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
        df['u_count'] = df['user_id'].map(df['user_id'].value_counts())
        df['i_count'] = df['item_id'].map(df['item_id'].value_counts())
        
        if binary:
            df['u_pop'] = 1/df.u_count
            # df['bitem_pop'] = 1/trainset.n_items
            # df['item_pop_count'] = 1/df.i_count
        else:
            df['nbu_pop'] = df.rating/df.u_count
            # df['nbitem_popc'] = df.rating/df.i_count
        
        df['item_pop'] = df.i_count/trainset.n_items
        df['trust'] = df.u_pop * df.item_pop
        # print(df)
        return df
        

    def set_ur_new(self, trainset, df_orgi):
        ur = defaultdict(list)
        for user_x, x_ur_list in trainset.ur.items():
            user_list = []
            df = cp.deepcopy(df_orgi)
            df = df.loc[df['user_id'] == user_x]
            for item, rating in x_ur_list:
                user_list.append((item, df.loc[df['item_id'] == item].trust.iat[0]))
            ur[user_x] = user_list
        self.trainset.ur = ur
        print('done set_ur_new')

    def set_ir_new(self, trainset, df_orgi):
        ir = defaultdict(list)
        for item_x, x_ir_list in trainset.ir.items():
            user_list = []
            df = cp.deepcopy(df_orgi)
            df = df.loc[df['item_id'] == item_x]
            for user, rating in x_ir_list:
                user_list.append((user, df.loc[df['user_id'] == user].trust.iat[0]))
            ir[item_x] = user_list
        self.trainset.ir = ir
        print('done set_ir_new')

    def single_user_trust(self, u, i):
        #only shows if it exist in the main
        df = cp.deepcopy(self.t_all_rating)
        try:
            df = df.loc[df['item_id'] == trainset.to_inner_uid(u)]
            val = df.loc[df['user_id'] == trainset.to_inner_iid(i)].trust
            # print(str(val))
            if not val.empty:
                return val.item()
            else:
                return 0
        except ValueError:  # user was not part of the trainset
            return 0
    
    def single_user_item_est_trust(self, u, i):
        est = 0  # <q_i, p_u>
        try:
            i = self.trainset.to_inner_iid(i)
            u = self.trainset.to_inner_uid(u)
            known_user = self.trainset.knows_user(u)
            known_item = self.trainset.knows_item(i)
            if known_user and known_item:
                    est = np.dot(self.qi[i], self.pu[u])
            return est
        except ValueError:  # user was not part of the trainset
            return 0

    def any_pair_trust(self, u, i, binary=True):
        df = cp.deepcopy(self.t_all_rating)
        try:
            u_count_df = df[df['user_id'] == self.trainset.to_inner_uid(u)]
            u_count = len(u_count_df.index)
            # print(u_count)
            i_count_df = df[df['item_id'] == self.trainset.to_inner_iid(i)]
            # print(i_count_df)
            i_count = len(i_count_df.index)
            # print(i_count)
            if binary:
                u_pop = 1/u_count
            item_pop = i_count/self.trainset.n_items
            trust = item_pop * u_pop
            return u_count, i_count, trust
        except ValueError:
            return 0, 0, 0

    def single_item_geti_count(self, i):
    	return self.trainset.ir[i]

    def single_item_trust_ir(self, u, i):
        try:
            for u,rating in trainset.ir[self.trainset.to_inner_iid(i)]:
                if u == self.trainset.to_inner_iid(u):
                    return rating
        except ValueError
            return 0
    def single_item_geti_count(self, i):
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
        if self.compare == 4:
            #trust is also used in estimation
            self.final_algo.pu = self.orginal_pu
            self.final_algo.qi = self.orginal_qi
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
trainset, testset = train_test_split(data,random_state=100, test_size=.2)
testset = trainset.build_testset()
algo_name = 'SVD_'
algo = SVD(n_factors= n_factors, random_state=100)
final_algo = SVD(n_factors= n_factors, random_state=100)
print('start')
myalgo = ImpliciTrust(trainset, testset, algo, final_algo, compare=compare, binary=binary)
myalgo.fit(trainset)
print(dataset_name)
print('compare ='+str(compare))
# myalgo.single_user_trust(543, 249)
# myalgo.single_item_trustlist(249)
print('myalgo.pu')
print(myalgo.pu)
print('myalgo.qi')
print(myalgo.qi)
print('binary')
print(str(binary))


predictions = myalgo.test(testset)

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
# df['trust'] = myalgo.single_user_trust(df.user_id, df.item_id)
t = []
est_t = []
i_count_list = []
u_count_list = []
pairtrust_list = []
# df['trust'] = df.apply(lambda row: myalgo.single_user_trust(row['user_id'], row['item_id']), axis=1)
for index, row in df.iterrows():
    val = myalgo.single_user_trust(row['user_id'], row['item_id'])
    esttr = myalgo.single_user_item_est_trust(row['user_id'], row['item_id'])
    ucount, icount, pairtrust = myalgo.any_pair_trust(row['user_id'], row['item_id'])
    t.append(val)
    i_count_list.append(icount)
    u_count_list.append(ucount)
    pairtrust_list.append(pairtrust)
    est_t.append(esttr)
df['trust'] = t
df['est_trust'] = est_t
df['u_count'] = u_count_list
df['i_count'] = i_count_list
df['pairtrust'] = pairtrust_list

print(df.head())
best_predictions = df.sort_values(by='ERROR IN PREDICTION')[:10]
print('best_predictions')
print(best_predictions)
worst_predictions = df.sort_values(by='ERROR IN PREDICTION')[-10:]
print('worst_predictions')
print(worst_predictions)

taining_predictions = df[df['trust'] > 0]
print('taining_predictions')
print(taining_predictions[:10])

print(dataset_name)
print('compare ='+str(compare))
print('binary')
print(str(binary))
# print(df.loc[df['user_id'] == '812'])

if save:
	dump('results/'+algo_name+dataset_name+'_n_factors_'+str(n_factors)+'_n_epochs_'+str(n_epochs)+'compare'+str(compare), predictions=predictions, algo=myalgo, verbose=1)

