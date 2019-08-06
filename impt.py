from surprise import AlgoBase
from collections import defaultdict
import copy as cp
from surprise.dump import dump
import pandas as pd
import numpy as np

import plotly.plotly as py


class ImpliciTrust(AlgoBase):

    def __init__(self, algo, final_algo, binary=True, compare=None, parellel=False, biased=True, random_state=100):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.final_algo = final_algo
        self.algo = algo
        self.binary = binary
        self.parellel = parellel
        self.random_state = random_state
        self.biased = biased
        self.compare = compare if compare is not None else 0
 
    def fit(self, trainset):
        print('fitting')
        self.trainset2 = cp.deepcopy(trainset)
        self.final_algo.fit(trainset)

        if (self.compare == 5) or (self.compare == 6):
            print('changing last col of final_algo qi n pu')
            '''
            add 1 to pu if trust dimension is 1xqi and add trust to qi
            else add 1s to qi and add trust to qu 
            otherwise add 1s trust and  trust n 1s to pu n qi respectively
            '''
            #just trust between users

            print('calculating trust for all DataFrame')
            self.t_all_rating = self.get_all_rating_trust(trainset, self.binary)
            self.trust_df = cp.deepcopy(self.t_all_rating)

            list_tu = self.trust_df.groupby(['user_id']).mean().trust.values.tolist()
            new_pu_list = []
            counter = 0
            for t2 in self.final_algo.pu:
                templist = []
                for x in t2:
                    templist.append(x)
                templist.append(list_tu[counter])
                if self.compare == 5:
                    templist.append(1)
                counter =+ 1
                new_pu_list.append(templist)
                # print(templist)
            self.final_algo.pu = new_pu_list
            self.algo.pu = self.trust_df.groupby(['user_id']).mean().trust.to_dict()

            list_qi = self.trust_df.groupby(['item_id']).mean().trust.values.tolist()
            new_qi_list = []
            counter = 0
            for t2 in self.final_algo.qi:
                templist = []
                for x in t2:
                    templist.append(x)
                    
                if self.compare == 5:
                    templist.append(1)
                templist.append(list_qi[counter])
                counter =+ 1
                new_qi_list.append(templist)
                # print(templist)
            self.final_algo.qi = new_qi_list
            self.algo.qi = self.trust_df.groupby(['item_id']).mean().trust.to_dict()

            print('self.final_algo.trainset.global_mean')
            print(self.final_algo.trainset.global_mean)

            
            if self.biased:
                bu = []
                bi = []
                yj = []
                for b in self.final_algo.bu:
                    bu.append(b)
                for b in self.final_algo.bi:
                    bi.append(b)
#                 print('self.final_algo.yj')
#                 print(self.final_algo.yj)
                for j in self.final_algo.yj:
#                     print('j')
#                     print(j)
                    k = [i for i in j]
#                     print('k')
                    if self.compare == 5:
                        k.append(0)
                    k.append(0)
#                     print(k)
                    yj.append(np.asarray(k))
                bu.append(0)
                if self.compare == 5:
                    bu.append(0)
                bi.append(0)
                if self.compare == 5:
                    bi.append(0)
                self.final_algo.bu = bu
                self.final_algo.bi = bi
                self.final_algo.yj = np.asarray(yj)
#                 print(yj)

            if self.compare == 5:
                self.final_algo.n_factors += 2
            else:
                self.final_algo.n_factors += 1
            self.trainset = self.final_algo.trainset
            self.yj = self.final_algo.yj
            self.bu = self.final_algo.bu
            self.bi = self.final_algo.bi

        elif (self.compare == 2) or (self.compare == 3) or (self.compare == 4) or (self.compare == 1):
            print('copying')
            self.orginal_pu = cp.deepcopy(self.final_algo.pu)
            self.orginal_qi = cp.deepcopy(self.final_algo.qi)
            print('calculating trust for all DataFrame')
            self.t_all_rating = self.get_all_rating_trust(self.trainset2, self.binary)
            print('setting trust values to trainset.ur n trainset.ir')
            trainset = self.set_ur_new(trainset, self.t_all_rating)
            trainset = self.set_ir_new(trainset, self.t_all_rating)
            print('fitting self.algo to create self.algo.pu n self.algo.qi (trust pu n qi)final fit')
            self.trainset = trainset
            self.algo.fit(self.trainset)
            print('done fitting with modified trainset')
            print('init complete')
            if self.compare == 2:
                self.final_algo.pu = self.orginal_pu * self.algo.pu
                self.final_algo.qi = self.orginal_qi * self.algo.qi
            if self.compare == 3:
                self.final_algo.pu = self.orginal_pu - self.algo.pu
                self.final_algo.qi = self.orginal_qi - self.algo.qi
            if self.compare == 4:
                print('conctanating final_algo with estimated trust')
                self.final_algo.pu = np.concatenate((self.orginal_pu, self.algo.pu))
                self.final_algo.qi = np.concatenate((self.orginal_qi, self.algo.qi))
                self.final_algo.n_factors *= 2
            if self.compare == 1:
                print(self.algo.pu)
                print(self.algo.qi)
                print('just using trust for predictions')
                self.final_algo.pu = self.algo.pu
                self.final_algo.qi = self.algo.qi
        
        elif self.compare == 0:
            print('baseline')
    

    def get_all_rating_trust(self, trainset, binary=True):
        df = pd.DataFrame(trainset.all_ratings(), columns=['user_id', 'item_id', 'rating'])
        df['u_count'] = df['user_id'].map(df['user_id'].value_counts())
        df['i_count'] = df['item_id'].map(df['item_id'].value_counts())
        
        df['u_pop'] = df.u_count/trainset.n_users
     
        
        df['item_pop'] = df.i_count/trainset.n_items

        
 
        df['trust'] =  df.u_pop * df.item_pop
    
        # df['trust'] = 1
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
        trainset.ur = ur
        print('done set_ur_new')
        return trainset

    def set_ir_new(self, trainset, df_orgi):
        ir = defaultdict(list)
        for item_x, x_ir_list in trainset.ir.items():
            user_list = []
            df = cp.deepcopy(df_orgi)
            df = df.loc[df['item_id'] == item_x]
            for user, rating in x_ir_list:
                user_list.append((user, df.loc[df['user_id'] == user].trust.iat[0]))
            ir[item_x] = user_list
        trainset.ir = ir
        print('done set_ir_new')
        return trainset

    def single_user_trust(self, u, i):
        #only shows if it exist in the main
        df = cp.deepcopy(self.t_all_rating)
        try:
            df = df[df['item_id'] == self.trainset.to_inner_uid(u)]
            val = df[df['user_id'] == self.trainset.to_inner_iid(i)].trust
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
#             known_user = self.trainset.knows_user(u)
#             known_item = self.trainset.knows_item(i)
#             if known_user and known_item:
            est = np.dot(self.algo.qi[i], self.algo.pu[u])
            return est
        except ValueError:  # user was not part of the trainset
            return 0
        
    def estimate(self, u, i):
        return self.final_algo.estimate(u,i)