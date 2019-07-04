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


class ImpliciTrust(AlgoBase):

    def __init__(self, trainset, testset, algo, final_algo, compare=None, binary=False, parellel=False, random_state=100):

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
        self.compare = compare if compare is not None else 0
 
    def fit(self, trainset):
    	print('fitting')
    	self.final_algo.fit(self.final_algo_trainset)
    	self.orginal_pu = cp.deepcopy(self.final_algo.pu)
    	self.orginal_qi = cp.deepcopy(self.final_algo.qi)
    	self.set_ur(trainset)
    	self.set_ir(trainset)
    	self.algo.fit(self.trainset)
    	self.pu = self.algo.pu
    	self.qi = self.algo.qi
    	print('done fitting with modified trainset')
    	
    def set_ur(self, trainset):
    	ir = defaultdict(list)
    	for item_x in range(trainset.n_items):
    		item_list = []
    		item_pop = len(trainset.ir[item_x])/trainset.n_items
    		for user, rating in trainset.ir[item_x]:
    			u_pop = 1/len(trainset.ur[user])
    			item_list.append((user,u_pop * item_pop))
    		ir[item_x] = item_list
    	self.trainset.ir = ir
    	print('done set_ur')

    def set_ir(self, trainset):
    	ur = defaultdict(list)
    	for user_x in range(trainset.n_users):
    		user_list = []
    		user_pop = 1/len(trainset.ur[user_x])
    		for item, rating in trainset.ur[user_x]:
    			item_pop = len(trainset.ir[item])/trainset.n_items
    			user_list.append((item, item_pop * user_pop))
    		ur[user_x] = user_list
    	self.trainset.ur = ur
    	print('done set_ir')

    def single_user_trustlist(self, u):
    	return self.trainset.ur[u]

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
compare = 3
n_factors = 20
n_epochs = 20
save = True
data = Dataset.load_builtin(dataset_name)
trainset, testset = train_test_split(data,random_state=100, test_size=.20)
algo_name = 'SVDpp_'
algo = SVDpp(random_state=100)
final_algo = SVDpp(random_state=100)
myalgo = ImpliciTrust(trainset, testset, algo, final_algo, compare=compare)
myalgo.fit(trainset)
print(dataset_name)
# myalgo.single_user_trustlist(543)
# myalgo.single_item_trustlist(249)

predictions = myalgo.test(testset)
for p in predictions[:100]:
		# print(t.ur[t.to_inner_uid(p.uid)])
		print(p)
accuracy.rmse(predictions)
if save:
	dump('results/'+algo_name+dataset_name+'_n_factors_'+str(n_factors)+'_n_epochs_'+str(n_epochs)+'compare'+str(compare), predictions=predictions, algo=myalgo, verbose=1)

