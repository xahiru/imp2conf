import pandas as pd
import numpy as np
from impt import ImpliciTrust
from impt import graph_by_py
import plotly.plotly as py
from surprise import Dataset
from surprise.model_selection import train_test_split

from surprise import SVD
from surprise import SVDpp
from surprise import SVDppimp
from surprise import accuracy

import copy as cp

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
# testset = trainset.build_testset()
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
# myalgo.t_all_rating[(myalgo.t_all_rating.user_id == 2) & (myalgo.t_all_rating.item_id==2)].trust.item()
def get_Iu(uid):
    """Return the number of items rated by given user"""
    try:
        return myalgo.trainset.to_inner_uid(uid)
    except ValueError:  # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """Return the number of users that have rated given item"""
    try:
        return myalgo.trainset.to_inner_iid(iid)
    except ValueError:  # item was not part of the trainset
        return 0 
df['Ii'] = df.user_id.apply(get_Iu)
df['Ui'] = df.item_id.apply(get_Ui)
df['error'] = abs(df.estimated_Ratings - df.ratings_ui)
t = []
est_t = []
i_count_list = []
u_count_list = []
pairtrust_list = []
ir_trust_list = []
for index, row in df.iterrows():
    val = myalgo.single_user_trust(row['user_id'], row['item_id'])
    esttr = myalgo.single_user_item_est_trust(row['user_id'], row['item_id'])
    # print(str(row['user_id'])+str(row['item_id']))
    ucount, icount, pairtrust = myalgo.any_pair_trust(row['user_id'], row['item_id'])
    ir_trust_list.append(myalgo.single_item_trust_ir_ur(row['user_id'], row['item_id']))
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
df['ir_trust'] = ir_trust_list

print(df.head())
best_predictions = df.sort_values(by='error')[:10]
print('best_predictions')
print(best_predictions)
worst_predictions = df.sort_values(by='error')[-10:]
print('worst_predictions')
print(worst_predictions)

taining_predictions = df[df['trust'] > 0]
print('taining_predictions')
print(taining_predictions.sort_values(by='error')[:10])

print(dataset_name)
print('compare ='+str(compare))
print('binary')
print(str(binary))


# graph_by_py(df)


if save:
	dump('results/'+algo_name+dataset_name+'_n_factors_'+str(n_factors)+'_n_epochs_'+str(n_epochs)+'compare'+str(compare), predictions=predictions, algo=myalgo, verbose=1)

