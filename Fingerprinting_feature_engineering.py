from rdkit import Chem
import numpy as np
from io import StringIO
import string
import re
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

import multiprocessing as mp
import random
import string
from rdkit.Chem import AllChem
import h5py

%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

def applyTo():
    while not inputq.empty():
        inp = inputq.get()
        smile = inp[0]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print smile
            print "fail"
            output.put((np.zeros((1,4096)),a))
        else:
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            fp = Chem.RDKFingerprint(mol,fpSize=4096)
            bs = np.fromstring(fp.ToBitString(),'u1') - ord('0')
            a = bs.reshape((1,4096))
            output.put((a,inp[1]))
    print "done"

	inputq = mp.Queue()
	output = mp.Queue()

def generateFeats(df_train)

	smdt = df_train
	
	if 'gap' in df_train:
		Y_train = smdt.gap.values
	else:
		Y_train = None
	smiles_tr = smdt.smiles

	smdt = smdt.drop(['gap'], axis=1)
	smdt = smdt.drop(['smiles'], axis=1)

	final_tr = np.zeros((len(smdt),4096))

	processes = [mp.Process(target=applyTo, args=()) for x in range(100)]

	i = 0

	while i < len(smiles_tr) and not inputq.full():
	    inputq.put([smiles_tr[i],i])
	    i += 1

	for p in processes:
	    p.start()

	b = 0
	print "0"
	while (not output.empty()) or (not inputq.empty()):
	    if b % 10000 == 0:
	        print b
	    b+=1
	    a = output.get()
	    final_tr[a[1]] = a[0]
	
	for p in processes:
	    p.terminate()
	
	return (np.hstack([smdt,final_tr]),Y_train)

def save(data,xname,yname):
	x_val, y_val, = generateFeats(data)
	h5f = h5py.File(name, 'w')
	h5f.create_dataset('dataset_1', data=x_val)
	h5f.close()
	h5f = h5py.File(yname, 'w')
	h5f.create_dataset('dataset_1', data=y_val)
	h5f.close()

def getdata():
	df_train = pd.read_csv("new_xtr_feat.csv")
	df_val = pd.read_csv("new_xval_feat.csv")
	df_test = pd.read_csv("new_xte_feat.csv")
	print "dataloaded"
	
	save(df_train,"/mnt/data/xtr.hd5","/mnt/data/ytrain.hd5")
	print "saved train"
	
	save(df_val,"/mnt/data/xval.hd5","/mnt/data/yval.hd5")
	print "saved val"
	save(df_test,"/mnt/data/xtest.hd5","/mnt/data/ytest.hd5")
	print "saved test"

def do_svd(train, test):
   svd = TruncatedSVD(n_components=200, random_state=42)
   svd=svd.fit(train)
   X=svd.transform(train)
   X_test=svd.transform(test)
   return X, X_test

def process(train,test):
	print "starting"
	X, X_test=do_svd(train[:899998,0:1024], test[:824231,0:1024])
	print "250 done"
	X2, X2_test=do_svd(train[:899998,1024:2048], test[:824231,1024:2048])
	print "500 done"
	X3, X3_test=do_svd(train[:899998,2048:3072], test[:824231,2048:3072])
	print "750 done"
	X4, X4_test=do_svd(train[:899998,3072:4097], test[:824231,3072:4097])
	print "1000 done"
	X_final=np.hstack((X,X2,X3,X4))
	X_te_final=np.hstack((X_test,X2_test,X3_test,X4_test))
	return (X_final,X_te_final)

def svd(train,test):
	X_final, X_te_final = process(train,test)
	h5f = h5py.File("x_tr_final.hd5", 'w')
	h5f.create_dataset("dataset_1", data=X_final)
	h5f.close()
	h5f = h5py.File("x_te_final.hd5", 'w')
	h5f.create_dataset("dataset_1", data=X_te_final)
	h5f.close()
	print "svd done"

def genstuff():
	h5f = h5py.File('/mnt/data/xtr.hd5','r')
	train = h5f['dataset_1'][:]
	h5f.close()
	h5f = h5py.File('/mnt/data/xtest.hd5','r')
	test = h5f['dataset_1'][:]
	h5f.close()
	svd(train,test)

def train(tn,Y_train):
	reg = xgb.XGBRegressor(nthread=30, learning_rate=0.02, n_estimators=2000, silent=0, seed=50,
	                           subsample= 0.8, colsample_bytree= 0.75, max_depth= 10)
	return reg.fit(tn,Y_train)

def regress():
	h5f = h5py.File('/mnt/data/x_tr_final.hd5','r')
	train = h5f['dataset_1'][:]
	h5f.close()
	h5f = h5py.File('/mnt/data/x_te_final.hd5','r')
	test = h5f['dataset_1'][:]
	h5f.close()
	h5f = h5py.File('/mnt/data/ytrain.hd5','r')
	y_train = h5f['dataset_1'][:]
	h5f.close()
	print "loaded again"
	model = train(train,y_train
	print "trained"
	joblib.dump(gbrt_fe, '/mnt/data/gbrt_fe.pkl') 
	
	y_hat=clf.predict(test)
	np.savetxt("done.csv", y_hat, delimiter=",")
	print "done"
	

getdata()
genstuff()
regress()

