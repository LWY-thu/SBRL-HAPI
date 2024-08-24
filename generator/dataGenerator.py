import numpy as np
import pandas as pd
import scipy.special
import csv
import sys
import os
from scipy.stats import norm
from scipy.stats import bernoulli    

from dataUtils import * 

class Syn_Generator_LWY(object):   
    def __init__(self, n,ate,sc,sh,one,depX,depU,
                 mV,mX,mU,mA,mXs,init_seed=4,seed_coef=10,details=0,
                 storage_path='./Data/', random_coef='F',use_one='F',data_version=1,dep=0,args=None):
        self.n = n # 数据集总量
        self.ate = ate
        self.sc = sc
        self.sh = sh
        self.depX = depX
        self.depU = depU
        self.one = one
        self.mV = mV # Dimensions of instrumental variables
        self.mX = mX # Dimensions of observed confounding variables
        self.mU = mU # Dimensions of unmeasured confounding variables
        self.mA = mA # Dimensions of adjustment variables
        self.mXs = mXs # Dimensions of spurious variables
        self.seed = init_seed
        self.seed_coef = seed_coef
        self.storage_path = storage_path
        self.data_version = data_version
        self.dep = dep
        self.dep1_coef = args.dep1_coef
        self.random_coef = random_coef
        self.use_one = use_one
        
        # coefs_t_VXU 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
        np.random.seed(1 * seed_coef * init_seed + 3)
        if random_coef == "True" or random_coef == "T":
            self.coefs_t_VXU = np.random.normal(size=mV + mX + mU)
        else:
            self.coefs_t_VXU = np.round(np.random.uniform(low=8, high=16, size=mV + mX + mU))
        if use_one == "True" or use_one == "T":
            self.coefs_t_VXU = np.ones(shape=mV + mX + mU)
    
        # coefs_y_XU: 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
        np.random.seed(2 * seed_coef * init_seed + 5)  # <--
        if random_coef == "True" or random_coef == "T":
            self.coefs_y_XU0 = np.random.normal(size=mX+mU+mA)
            self.coefs_y_XU1 = np.random.normal(size=mX+mU+mA)
        else:
            self.coefs_y_XU0 = np.round(np.random.uniform(low=8, high=16, size=mX + mU+mA))
            self.coefs_y_XU1 = np.round(np.random.uniform(low=8, high=16, size=mX + mU+mA))
        if use_one == "True" or use_one == "T":
            self.coefs_y_XU0 = np.ones(shape=mX+mU+mA)
            self.coefs_y_XU1 = np.ones(shape=mX+mU+mA)
            

        self.set_path(details)
        
        with open(self.data_path+'coefs.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(self.coefs_t_VXU)
            csv_writer.writerow(self.coefs_y_XU0)
            csv_writer.writerow(self.coefs_y_XU1)
        
    def get_multivariate_normal_params(self, dep, m, seed=0):
        np.random.seed(seed)
        if dep == 1:
            print('dep1')
            mu = np.random.normal(size=m) / 10.
            ''' sample random positive semi-definite matrix for cov '''
            temp = np.random.uniform(size=(m, m))
            temp = .5 * (np.transpose(temp) + temp)
            sig = (temp + m * np.eye(m)) / self.dep1_coef
        elif dep == 2:
            print('dep2')
            mV = self.mV
            mX = self.mX + self.mXs
            mU = self.mU + self.mA
            # mA = self.mA
            depU = self.depU
            depX = self.depX

            mu = np.zeros(m)
            sig = np.eye(m)
            temp_sig = np.ones(shape=(m-mV,m-mV))
            temp_sig = temp_sig * depU
            sig[mV:,mV:] = temp_sig

            sig_temp = np.ones(shape=(mX,mX)) * depX
            sig[mV:-mU,mV:-mU] = sig_temp

            sig[np.diag_indices_from(sig)] = 1
        else:
            print('dep0')
            mu = np.zeros(m)
            sig = np.eye(m)

        return mu, sig

    def get_latent(self, n, m, dep, seed):
        L = np.array((n*[[]]))
        if m != 0:
            mu, sig = self.get_multivariate_normal_params(dep, m, seed)
            L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
            data_path = self.data_path
            with open(data_path+'info/mu.csv', 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                csv_writer.writerow(mu)
            with open(data_path+'info/sig.csv', 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                for row in sig:
                    csv_writer.writerow(row)
        return L

    def set_path(self,details):
        which_benchmark = f'SynSBRL{self.data_version}_ICUAV_biny_'+'_'.join(str(item) for item in [self.sc, self.sh, self.one, self.depX, self.depU,self.VX])
        print(which_benchmark)
        data_path = self.storage_path+'/data/'+which_benchmark
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        which_dataset = '_'.join(str(item) for item in [self.mV, self.mX, self.mU, self.mA, self.mXs])
        data_path += '/'+which_dataset+'/'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self.data_path = data_path
        self.which_benchmark = which_benchmark
        self.which_dataset = which_dataset

        if details:
            print('#'*30)
            print("syn_data_generator_v2")
            print('The data path is: {}'.format(self.data_path))
            print('The ATE:')
            print('-'*30)
            print(f'ate: {1+self.ate}')  
            print('-'*30)
        
    def run(self, n=None, num_reps=10):
        self.num_reps = num_reps
        
        seed_coef = self.seed_coef 
        init_seed = self.seed

        if n is None:
            n = self.n

        print('Next, run dataGenerator: ')
        data_path = self.data_path 
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs(os.path.dirname(data_path+'info/'), exist_ok=True)

        train_dict, train_df = self.get_data(n=n, seed=3*seed_coef*init_seed+777)  
        train_df.to_csv(data_path + '/raw.csv', index=False)

        ''' bias rate '''
        br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0]
        brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30'}

        dim_x = self.mX + self.mXs
        dim_v = self.mV
        dim_a = self.mA
        size_train = n
        size_test = n
        for r in br:
            conty_train_data = {'x':np.zeros((size_train, dim_x, num_reps)), 'v':np.zeros((size_train, dim_v, num_reps)), 
                                'a':np.zeros((size_train, dim_a, num_reps)), 
                                't':np.zeros((size_train, num_reps)), 
                                'yf':np.zeros((size_train, num_reps)), 'ycf':np.zeros((size_train, num_reps)), 'mu0':np.zeros((size_train, num_reps)), 'mu1':np.zeros((size_train, num_reps))}
            conty_test_data = {'x':np.zeros((size_test, dim_x, num_reps)), 'v':np.zeros((size_test, dim_v, num_reps)), 
                                'a':np.zeros((size_train, dim_a, num_reps)),
                               't':np.zeros((size_test, num_reps)),
                               'yf':np.zeros((size_test, num_reps)), 'ycf':np.zeros((size_test, num_reps)), 'mu0':np.zeros((size_test, num_reps)), 'mu1':np.zeros((size_test, num_reps))}

            for exp in range(num_reps):
                print(f'Run {exp}/{num_reps}:{brdc[r]} ')
                train_df_ood = correlation_sample(train_df, r, n, self.mXs, seed=3*seed_coef*init_seed+exp+777)
                val_df_ood = correlation_sample(train_df, r, n, self.mXs, seed=4*seed_coef*init_seed+exp+777)
                test_df_ood = correlation_sample(train_df, r, n, self.mXs, seed=5*seed_coef*init_seed+exp+777)

                path = data_path + '/{}/'.format(exp)
                os.makedirs(os.path.dirname(path + f'ood_{brdc[r]}/'), exist_ok=True)

                train_df_ood.to_csv(path + f'ood_{brdc[r]}/train.csv', index=False)
                val_df_ood.to_csv(path + f'ood_{brdc[r]}/val.csv', index=False)
                test_df_ood.to_csv(path + f'ood_{brdc[r]}/test.csv', index=False)

                train_tmp_sample = train_df_ood.sample(n=size_train)
                test_tmp_sample = test_df_ood.sample(n=size_test)

                for k in conty_train_data.keys():
                    if k=='x' or k == 'v' or k == 'a':
                        conty_train_data[k][:,:,exp] = get_var_df(train_tmp_sample, k)
                        conty_test_data[k][:,:,exp] = get_var_df(test_tmp_sample, k)
                    elif k=='yf':
                        conty_train_data[k][:,exp] = get_var_df(train_tmp_sample, 'y').squeeze()
                        conty_test_data[k][:,exp] = get_var_df(test_tmp_sample, 'y').squeeze()
                    elif k=='ycf':
                        conty_train_data[k][:,exp] = get_var_df(train_tmp_sample, 'f').squeeze()
                        conty_test_data[k][:,exp] = get_var_df(test_tmp_sample, 'f').squeeze()
                    else:
                        conty_train_data[k][:,exp] = get_var_df(train_tmp_sample, k).squeeze()
                        conty_test_data[k][:,exp] = get_var_df(test_tmp_sample, k).squeeze()
            # data saving
            np.savez(data_path+f'{self.data_version}_r{brdc[r]}.train.npz', **conty_train_data)
            np.savez(data_path+f'{self.data_version}_r{brdc[r]}.test.npz', **conty_test_data)

                      
        print('-'*30)


    def get_data(self, n, seed):

        np.random.seed(1*seed)
        mV = self.mV
        mX = self.mX
        mU = self.mU
        mA = self.mA
        mXs = self.mXs
        random_coef = self.random_coef
        init_seed = self.seed
        seed_coef = self.seed_coef

        # harder datasets
        dep = self.dep  # overwright; dep=0 generates harder datasets

        # Big Dataset size for sample
        n_trn = n * 100

        # all dimension
        max_dim = mV + mX + mU + mA + mXs

        # Variables
        temp = self.get_latent(n=n_trn, m=max_dim, dep=dep, seed=seed + 4)
              
        # Divide V X U A Xs
        V = temp[:, 0:mV]
        X = temp[:, mV:mV+mX]
        U = temp[:, mV+mX:mV+mX+mU]
        A = temp[:, mV+mX+mU:mV+mX+mU+mA]
        Xs = temp[:, mV+mX+mU+mA:mV+mX+mU+mA+mXs]
        X_all = np.concatenate((X, Xs), 1)
        T_vars = np.concatenate([V,X,U], axis=1) # T_vars: variable related T
        Y_vars = np.concatenate([X,U,A], axis=1) # XU: variable related Y
        
        # generate Treatment
        np.random.seed(2*seed)
        z = np.dot(T_vars, self.coefs_t_VXU)
        per = np.random.normal(size=n_trn)
        pi0_t1 = scipy.special.expit( self.sc*(z+self.sh+per) )
        t = bernoulli.rvs(pi0_t1)
        
        # generate ATE
        coef_devide_2 = 10
        coef_devide_3 = 10    	  
        if self.random_coef == "True" or self.random_coef == "T" or self.use_one == "True" or self.use_one == "T":               
            mu_0 = np.dot(Y_vars**1, self.coefs_y_XU0) / (mX+mU+mA)
            mu_1 = np.dot(Y_vars**2, self.coefs_y_XU1) / (mX+mU+mA) + self.ate
        else:
            mu_0 = np.dot(Y_vars**1, self.coefs_y_XU0) / (mX+mU+mA) / coef_devide_2
            mu_1 = np.dot(Y_vars**2, self.coefs_y_XU1) / ((mX+mU+mA) + self.ate)/ coef_devide_3

        # generate Y
        np.random.seed(3*seed)
        y = np.zeros((n_trn, 2))
        y[:,0] = mu_0 + np.random.normal(loc=0., scale=.01, size=n_trn)
        y[:,1] = mu_1 + np.random.normal(loc=0., scale=.01, size=n_trn)

        yf = np.zeros(n_trn)
        ycf = np.zeros(n_trn)
        for i, t_i in enumerate(t):
            yf = np.append(yf, y[i, int(t_i)])
            ycf = np.append(ycf, y[i, int(1-t_i)])

        data_dict = {'V':V, 'A':A,'U':U, 'X':X, 'Xs':Xs,'z':z, 'pi':pi0_t1, 't':t, 'mu0':mu_0, 'mu1':mu_1, 'yf':yf, 'y':y, 'ycf':ycf}
        data_all = np.concatenate([V, X, U, A, Xs, z.reshape(-1,1), pi0_t1.reshape(-1,1), t.reshape(-1,1), mu_0.reshape(-1,1), mu_1.reshape(-1,1), yf.reshape(-1,1), ycf.reshape(-1,1)], axis=1)
        data_df = pd.DataFrame(data_all,
                               columns=['v{}'.format(i+1) for i in range(V.shape[1])] + 
                               ['x{}'.format(i+1) for i in range(X.shape[1])] + 
                               ['u{}'.format(i+1) for i in range(U.shape[1])] + 
                               ['a{}'.format(i+1) for i in range(A.shape[1])] + 
                               ['xs{}'.format(i+1) for i in range(Xs.shape[1])] + 
                               ['z','pi','t','mu0','mu1','y','f'])
        
        return data_dict, data_df
    

