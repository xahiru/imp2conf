from surprise.dump import dump
from surprise.dump import load
from surprise import accuracy

predictions, algo = load('results/myalgosvdml-100k_n_factors_20_n_epochs_20_beta_0.5694444444444444_myalgo_True_pminusq_True_ptimesq_False')
# print(algo.qi)
# print(algo.pu)
# 80000
mae = accuracy.mae(predictions)
print(len(predictions))
rc = len(predictions)/80000
print('Rating Coverage (RC) ='+ str(rc))
F1 = (2*mae*rc)/(mae+rc)
print('F1 ='+ str(F1))