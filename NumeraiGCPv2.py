import numpy as np
#import pyarrow.parquet as pq
import eli5
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from eli5.sklearn import PermutationImportance
#rom lightgbm import LGBMRegressor as lgb,plot_importance
from catboost import CatBoostRegressor
from numerapi import NumerAPI
from google.colab import drive,files
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture as GMM
from xgboost import XGBRegressor as xgbr
import keras_tuner
from skopt import BayesSearchCV
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn import linear_model
import pickle
import cloudpickle
import gc
from scipy.stats import stats
import json
#import UMAP
counter = 0
import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.base import BaseEstimator, ClassifierMixin
# purged embargo splitcv method
# K-Fold presumes iid over split which is not true for time-series
#

class PurgedTimeSeriesSplitGroups(_BaseKFold):
    def __init__(self,groups, n_splits=5, purge_groups=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.purge_groups = purge_groups
        self.groups = groups

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        groups = self.groups
        n_samples = _num_samples(X)
        n_folds = self.n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds + self.purge_groups > n_groups:
            raise ValueError((f"Cannot have number of folds plus purged groups "
                              f"={n_folds+self.purge_groups} greater than the "
                              f"number of groups: {n_groups}."))
        indices = np.arange(n_samples)
        test_size = ((n_groups-self.purge_groups) // n_folds)
        test_starts = [n_groups-test_size*c for c in range(n_folds-1, 0, -1)]
        for test_start in test_starts:
            yield (indices[groups.isin(group_list[:test_start-self.purge_groups])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])
            


class WeightedSumModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_weight_pairs):
        self.model_weight_pairs = model_weight_pairs

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        weighted_sum_preds = np.zeros(X.shape[0])
        for model, weight in self.model_weight_pairs:
            lambda_predict = lambda data: weight * model.predict(data)
            weighted_sum_preds += lambda_predict(X)
        return weighted_sum_preds
    


class PiEnsembleModel():

    def __init__(self,public_id,secret_key,p):

        #drive.mount('/NumeraiGCPreal')

        self.napi = NumerAPI(public_id= public_id, secret_key =secret_key)

        self.curr_round = self.napi.get_current_round()
        #self.curr_round = 470

        self.p = p

        self.cat_col ='era'

        self.ms = self.p -1

        self.drop_feat_imp1  = None

        self.drop_feat_imp2 = None

        self.max_eval = 20

        self.riskyF = None
        print(self.curr_round)
       # dataset_name = 'v3'

      #  self.napi.download_current_dataset(unzip=True)

        #current_ds = self.napi.get_current_round()
       # self.latest_round = os.path.join('numerai_dataset_'+str(self.curr_round))

       # training_data = pd.read_csv(os.path.join(self.latest_round, "numerai_training_data.csv")).set_index("id")
# The tournament data is the data that Numerai uses to evaluate your model.
     #   tournament_data = pd.read_csv(os.path.join(self.latest_round, "numerai_tournament_data.csv")).set_index("id")
     #   print(tournament_data.data_type.unique())

     #   example_preds = pd.read_csv(os.path.join(latest_round, "example_predictions.csv")).set_index("id")

     #   validation_data = tournament_data[tournament_data.data_type == "validation"]


        #self.napi.download_dataset(f"{dataset_name}/train.parquet")
        # self.napi.download_dataset(f"{dataset_name}/validation.parquet")
        # self.napi.download_dataset(f"{dataset_name}/live.parquet", f"{dataset_name}/live_{self.curr_round}.parquet")

       # print(training_data.head())


    def Init(self,trainvalapp= True, test_app= False):

        def initpipeline(X,tr = False):

            X[self.cat_col] = LabelEncoder().fit_transform(X[self.cat_col]).astype('int64')

            num_col = X.drop([self.cat_col,'data_type'],axis = 1).columns

            # massively reduce memory

            X[num_col] = X[num_col].astype('float16')

        # take every fourth
            #if tr:

                #eraf = np.arange(self.ms,self.X_era[self.cat_col].max(),4)
               # print(eraf)
                #X =  X[ X[self.cat_col].isin(eraf)]

        # take upper half of more recent data ram threshold on colab

         #   X =  X.iloc[int( X.shape[0]/2): X.shape[0]]

          #  y = X['target']
            # remove useless feat

            X= X.drop('data_type', axis =1 )

#            print(self.X['Era'].unique())

        #    idx = X.reset_index()['id']

            return X

        def liveprep(live):

           # live[self.cat_col]=LabelEncoder().fit_transform(live[self.cat_col]).astype('int64')

            num_col = live.columns

            live[num_col] = live[num_col].astype('float16')

           # live = live.drop('data_type',axis =1 )

           # lid = live.reset_index()['id']

            return live


        if trainvalapp:

            #self.X_era = pd.read_parquet('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round) + '_v3_numerai_training_data.parquet')
           # self.X_era =training_data = pd.read_csv(os.path.join(self.latest_round, "numerai_training_data.csv")).set_index("id")
            self.napi.download_dataset("v4.1/train.parquet");
            self.napi.download_dataset("v4.1/features.json");

            feature_metadata = json.load(open("v4.1/features.json"))
            feature_cols = feature_metadata["feature_sets"]["small"]


            self.X_era = pd.read_parquet("v4.1/train.parquet", columns= ["era"] + feature_cols + ["target"]+["data_type"])
            #self.X_era=self.X_era.iloc[int( self.X_era.shape[0]/2): self.X_era.shape[0],:]

            self.X_era  = initpipeline(self.X_era,tr=True)

            self.X_era.to_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round)+'highera'+str(self.p)+'.csv')

            #self.X_v = pd.read_parquet('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round) + '_v3_numerai_validation_data.parquet')
           # self.X_v = pd.read_csv(os.path.join(self.latest_round, "numerai_tournament_data.csv")).set_index("id")
          #  self.X_v= self.X_v[self.X_v.data_type == "validation"]

# ram criteria can handle half of validation and training

           # self.X_v = self.X_v.iloc[int(self.X_v.shape[0]/2):self.X_v.shape[0]]
           # self.napi.download_dataset("v4.1/validation.parquet");

# Load the validation data but only the "small" subset of features
           # self.X_v = pd.read_parquet("v4.1/validation.parquet", columns=["era", "data_type"] + feature_cols + ["target"])

            # Filter for data_type == "validation"
           # self.X_v= self.X_v[self.X_v["data_type"] == "validation"]


          #  self.X_v = initpipeline(self.X_v)

          #  self.X_v.to_csv('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')



        elif test_app:

            #self.X_live = pd.read_parquet('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round) + '_v3_numerai_live_data.parquet')
          #  self.X_live = pd.read_csv(os.path.join(self.latest_round, "numerai_tournament_data.csv")).set_index("id")
            self.napi.download_dataset("v4.1/live.parquet", f"v4/live_{self.curr_round}.parquet");
            self.napi.download_dataset("v4.1/features.json");
            feature_metadata = json.load(open("v4.1/features.json"))
            feature_cols = feature_metadata["feature_sets"]["small"]
# Load live features
            self.X_live = pd.read_parquet(f"v4/live_{self.curr_round}.parquet", columns=feature_cols)

           # self.X_live= self.X_live[self.X_live.data_type == "live"]

            self.X_live= liveprep(self.X_live)

            self.X_live.to_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round)+'liveera'+str(1)+'.csv')

       # self.X_era.head()

    @staticmethod
    def OLSNeut(endog,preds,proportion=0.15) -> pd.DataFrame():

        scores = preds
        exposures = endog.values
        lm = linear_model.Lasso(alpha = 10)
        lm.fit(exposures)
        lm_sample = lm.predict(exposures)
        scores = scores - proportion * lm_sample.reshape(len(scores),1)
        result = pd.DataFrame(scores)

        return result



    def LoadData(self, trainvalapp=True, test_app = False,trainvalapproll=False,poorera= False,cluster_trainvalapp=False):


        if trainvalapp:

            self.X_era = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round-1)+'highera'+str(self.p)+'.csv')

       #     self.X_era= self.X_era.iloc[0:200,:]

            self.y_t = self.X_era['target']

        #self.X_era = self.X_era.dropna()

            #self.X_v = pd.read_csv('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')

          #  self.X_v = self.X_v.iloc[0:100,:]

            #self.y_v = self.X_v['target']

            #self.v_id = self.X_v['id']


            self.targ = [ c for c in self.X_era.columns if c.startswith('target')]


          #  self.X_newera= pd.concat([self.X_era,self.X_v]).sort_values(by='era',ascending =True)
         #   self.y_newt= self.X_newera['target']
            self.X_era= self.X_era.drop(self.targ,axis =1)
            #self.X_v = self.X_v.drop(self.targ,axis = 1)

            self.X_era = self.X_era.drop('id',axis =1 )
            #self.X_v= self.X_v.drop('id',axis=1)
           #save ram

            ue= self.X_era['era'].unique()
            #uv = self.X_v['era'].unique()

            print('unique X_era {}'.format(ue))
            #print('unique X_v {}'.format(uv))

            #print(' union of uniques {}'.format(np.intersect1d(ue,uv)))

            self.X_live = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round-1)+'liveera'+str(1)+'.csv')

            self.t_id = self.X_live.reset_index()['id']

            self.X_live= self.X_live.drop('id',axis=1)



        elif trainvalapproll:


            self.X_era = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round)+'highera'+str(self.p)+'.csv')

       #     self.X_era= self.X_era.iloc[0:200,:]

           # self.y_t = self.X_era['target']

        #self.X_era = self.X_era.dropna()

        #    self.X_v = pd.read_csv('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')

          #  self.X_v = self.X_v.iloc[0:100,:]

        #    self.y_v = self.X_v['target']

       #     self.v_id = self.X_v['id']


          #  targ = [ c for c in self.X_era.columns if c.startswith('target')]


          #  self.X_newera= pd.concat([self.X_era,self.X_v]).sort_values(by='era',ascending =True)
         #   self.y_newt= self.X_newera['target']
            #self.X_era= self.X_era.drop(targ,axis =1)
            #self.X_v = self.X_v.drop(targ,axis = 1)

            self.X_era = self.X_era.drop('id',axis =1 )
        #    self.X_v= self.X_v.drop('id',axis=1)
           #save ram

            ue= self.X_era['era'].unique()
       #     uv = self.X_v['era'].unique()

            print('unique X_era {}'.format(ue))
         #   print('unique X_v {}'.format(uv))

       #     print(' union of uniques {}'.format(np.intersect1d(ue,uv)))
          #  self.X_newera= self.X_newera.drop(['target','id'],axis=1)
            # remove these frames from memory
            #print('neut')
            #print(self.Neut(self.X_era,self.y_t))

       #     del [[self.X_era,self.X_v]]
        #    gc.collect()
        #    self.X_era=pd.DataFrame()
           # self.X_v=pd.DataFrame()

      #  if cluster_trainvalapp:

      #      self.X_era = pd.read_csv('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round)+'highera'+str(self.p)+'.csv')

      #      self.X_era=self.X_era.iloc[int( self.X_era.shape[0]/2) + int( self.X_era.shape[0]/3): self.X_era.shape[0],:]

      #      self.y_t = self.X_era['target']



        #self.X_era = self.X_era.dropna()

      #      self.X_v = pd.read_csv('/content/gdrive/My Drive/f0/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')

          #  self.X_v = self.X_v.iloc[0:100,:]

      #      self.y_v = self.X_v['target']

       #     self.v_id = self.X_v['id']


      #      targ = [ c for c in self.X_era.columns if c.startswith('target')]


          #  self.X_newera= pd.concat([self.X_era,self.X_v]).sort_values(by='era',ascending =True)
      #   #   self.y_newt= self.X_newera['target']
      #      self.X_era= self.X_era.drop(targ,axis =1)
          #  self.X_v = self.X_v.drop(targ,axis = 1)

      #      self.X_era = self.X_era.drop('id',axis =1 )
         #   self.X_v= self.X_v.drop('id',axis=1)
           #save ram
       #     targ =[]


     #  print(self.X_era.shape[0])

     #   print(self.X_v.shape[0])


        elif poorera:
            #self.curr_round = 449

            self.X_era = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round)+'highera'+str(self.p)+'.csv')

            self.X_v = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')




       #     self.X_era= self.X_era.iloc[0:200,:]

            #self.y_t = self.X_era['target']



        #self.X_era = self.X_era.dropna()

            #self.X_v = pd.read_csv('/content/drive/My Drive/CURRDATA/'+str(self.curr_round)+'vera'+str(1)+'.csv')

          #  self.X_v = self.X_v.iloc[0:100,:]

            #self.y_v = self.X_v['target']

            #self.v_id = self.X_v['id']
        # test neut func
     #   print('neut test {}'.format(self.NormNeutPred(self.X_v, self.X_v.columns, self.y_v.values.reshape(-1,1),0.4)))
       # print(len(np.intersect1d(self.X_live.columns, self.X_era.columns)))
        return self

    def DeriveMonoconstraint(self):

        self.corrs=np.ones(self.X_era.shape[1])
        for ind, c in enumerate(self.X_era.columns):
            print(ind)
            corr = np.corrcoef(self.y_t.values.reshape(-1,1),self.X_era[c].values.reshape(-1,1))[0,1]
            if corr > 0:
                pass
            elif corr <  0:
                self.corrs[ind] = -1


        print(self.corrs)


        return self


    def PermutationImp(self,model,perc = 0.95):

        self.y_t = self.X_era['target']

        self.X_era = self.X_era.drop(['target','era','eraB'], axis = 1 )


        #cols = self.X_era.columns

        #self.X_v = self.X_v[cols]

        perminst = PermutationImportance(model)

        perminst.fit(self.X_era,self.y_t)

        filter = SelectFromModel(perminst,threshold = 0.05,prefit=True)

        ft_imp2 = perminst.feature_importances_

        sort_imp_ind = np.argsort(-ft_imp2)

        sorted_ft_name  = self.X_era.columns[sort_imp_ind]

        ft_name1= sorted_ft_name[0:int(perc * len(sorted_ft_name))]

        self.drop_feat_imp1 = self.X_era.drop(ft_name1, axis=1 ).columns

        #self.X_train = self.X_train.drop(self.drop_feat_imp2,axis =1 )

        #self.X_test = self.X_test.drop(self.drop_feat_imp2,axis =1 )

        return self.drop_feat_imp1


    def FeatSelectMef(self,gridg,n_splits =3 ,perc = 0.95):

        st_time = time.time()

   #     n_neighbors = 15
   #     min_dist = 0
   #     n_components = 60

  #      umap= UMAP(n_neighbors= n_neighbors,min_dist =min_dist, n_components = n_components)
#
    #    dt = umap.fit_transform(self.X_era)

    #    umap_feat= [ f for f in range(dt.shape[1])]

        colorplot ='skyblue'

        initg = {
        "n_estimators" : 150,
      #  "max_depth" : 5,
        "learning_rate" : 0.01,
        "eval_metric":"rmse",
     #  'early_stopping_rounds' : 5,
      #  "feature_selector":'greedy'
     #   "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
        }

        model_g = xgbr(**initg)
     #   sub_tss= KFold(n_splits = 5 )
        ptg = PurgedTimeSeriesSplitGroups(self.X_era[self.cat_col], n_splits=n_splits)


        bscv = GridSearchCV(model_g,gridg, cv = ptg)

        bscv.fit(self.X_era,self.y_t)

     #   def score(gridg):
     #
     #       model = xgbr(**initg)
      #      model.fit(self.X_era,self.y_t,eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)]
      #      ,verbose=False)
    #
     #       Y_pred = model.predict(self.X_v)
     #       score = np.sqrt(mean_squared_error(self.y_v, Y_pred))
            #print(score)
      #      return {'loss': score, 'status': STATUS_OK}

      #  def opt(trials,gridg):

      #      best = fmin(score,gridg,algo = tpe.suggest, max_evals = 20)

      #      return best


      #  trials = Trials()

        #optparams = opt(trials,gridg)
        optparams = bscv.best_params_

        # apply result of hyperopt hyperparameter tuning
       # lg_fu = xgbr(**initg)

        model_g.set_params(**optparams)

        model_g.fit(self.X_era,self.y_t, eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)])
   #     ft_name = self.X_era.columns

        ft_imp= model_g.feature_importances_


        sort_imp_ind = np.argsort(-ft_imp)

        sorted_ft_name  = self.X_era.columns[sort_imp_ind]

        ft_name1= sorted_ft_name[0:int(perc * len(sorted_ft_name))]

        self.drop_feat_imp1 = self.X_era.drop(ft_name1, axis=1 ).columns

        drop_feat= pd.DataFrame()
#        drop_feat2= pd.DataFrame()
        drop_feat['drop_feat1']= self.drop_feat_imp1
 #       drop_feat2['drop_feat2']= self.drop_feat_imp2
        drop_feat.to_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv',encoding='utf-8', index=False)
  #      drop_feat2.to_csv('/content/drive/My Drive/drop_feat2.csv',encoding='utf-8', index=False)
        #plot_importance(model_g,fig_size=(160,100),color=colorplot)


        st_time2 = time.time()
        print('Time taken for MDA {}'.format(st_time2-st_time))


       # es = shap.Explainer(model_g)
      #  shap_v = es(self.X_era)

     #   self.drop_feat_imp2= self.X_era.drop(shap_v.columns,axis =1)
     #   drop_feat2= pd.DataFrame()
     #   drop_feat2['drop_feat1']= self.drop_feat_imp2
     ##   drop_feat.to_csv('/content/drive/My Drive/CURRDATA/SHAPdrop_feat'+str(self.ms)+'.csv',encoding='utf-8', index=False)

        return self


    def RollingFeatSelectMef(self,n_splits =3 ,perc = 0.95):

        initg = {
      #  "n_estimators" :125,
        "max_depth" : 8,
        #"learning_rate" : 0.01,
        "eval_metric":"rmse",
     #   "warm_start":True
        #"early_stopping_rounds": 2,
        #"feature_selector":'greedy'
        #"colsample_bytree" : 0.1,
        #"tree_method" : 'gpu_hist'
        }


        gridg = {

        #'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
        'eta': hp.quniform('eta', 0.001, 0.09, 0.001),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        #'max_depth':  hp.choice('max_depth', np.arange(4, 7, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight',range(1,10,1)),
        #'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 3, 0.5),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.8, 1, 0.05),
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        #'nthread': 4,
          #  'booster': 'gbtree',
        #'tree_method': 'exact',
        #'silent': 1,
        #
        }

        # opt over out of sample time series single slice
        # can loop over windows to compare


        spread = 0
        self.X_era['eraB'] = np.arange(0,self.X_era.shape[0],1)
        copx = self.X_era.copy()
        #copxv= self.X_v.copy()
        mse= 0
        frwindow = 1
        ue= self.X_era['era'].unique()
      #  uv = self.X_v['era'].unique()
        erawindows = ue
        #erawindows = np.intersect1d(ue,uv)
        targ = [ c for c in self.X_era.columns if c.startswith('target')]

        #self.y_v = self.X_v['target']


       # self.X_v = self.X_v.drop(targ,axis = 1)

        def RollingOptimizeMSE(params):

            model_m = xgbr(**initg)
            model_m.set_params(**params)
            mse = 0


                # rolling sliding winodow over era

            for col, er in enumerate(erawindows):


                self.X_era = self.X_era[ self.X_era['era'] == er ]

            #    self.X_v = self.X_v[self.X_v['era']== er]

                self.y_t = self.X_era['target']


             #   self.y_v = self.X_v['target']

                self.X_era= self.X_era.drop(targ,axis =1)

                ptg = PurgedTimeSeriesSplitGroups(groups = self.X_era['eraB'])

                for tn, (trainingrow, testingrow) in enumerate(ptg.split(self.X_era,y= self.y_t)):
         #      self.X_v = self.X_v.drop(targ,axis = 1)

                    #self.X_era = pd.concat([self.X_era.iloc[roll:,:],self.X_v.iloc[0:roll,:]],axis=0)

                    #self.y_t= pd.concat([self.y_t[roll:],self.y_v[0:roll]],axis = 0 )
                    # consider the last split over the era
                    if tn==4:

                        model_m.fit(self.X_era.drop(['era','eraB'],axis=1).iloc[trainingrow,:].values,self.y_t.values[trainingrow].reshape(-1,1).ravel())

                        predth = model_m.predict(self.X_era.drop(['era','eraB'],axis=1).iloc[testingrow,:].values)

                        mse += mean_squared_error(self.y_t.values[testingrow].reshape(-1,1), predth)

                        #model_m.n_estimators += initg['n_estimators']

                        self.X_era= copx

                    else:

                        pass


         #       self.X_v = copxv


            lossh = mse


            return {'loss':lossh,'status':STATUS_OK}



        trials = Trials()
        optparamsbayes = fmin(RollingOptimizeMSE,space = gridg,algo=tpe.suggest,max_evals=1,trials=trials)
        optparams = optparamsbayes

        ue= self.X_era['era'].unique()
      #  uv = self.X_v['era'].unique()
        erawindows = ue
       # erawindows = np.intersect1d(ue,uv)
        targ = [ c for c in self.X_era.columns if c.startswith('target')]

        model_g = xgbr(**initg)
        model_g.set_params(**optparams)


        for er in erawindows:

            self.X_era = self.X_era[ self.X_era['era'] == er ]

           # self.X_v = self.X_v[self.X_v['era']== er]

            self.y_t = self.X_era['target']


       #    self.y_v = self.X_v['target']

            self.X_era= self.X_era.drop(targ,axis =1)
      #      self.X_v = self.X_v.drop(targ,axis = 1)

                # bscv = GridSearchCV(model_g,gridg, cv = ptg)

                #    bscv.fit(self.X_era,self.y_t)


                #    optparams = bscv.best_params_


            model_g.fit(self.X_era.drop(['era','eraB'],axis=1),self.y_t,eval_set = [(self.X_era.drop(['era','eraB'],axis=1),self.y_t)])



            print("for XGBoostRegressor @ {} Fold/Model the Training Corr is {}".format(self.ms,self.CorrelationScore(self.y_t,model_g.predict(self.X_era.drop(['era','eraB'],axis=1)))))


          #  print("for XGBoostRegressor @ {} Fold/Model the Validation Corr is {}".format(self.ms,self.CorrelationScore(self.y_v,model_g.predict(self.X_v))))
            #model_g.n_estimators += initg['n_estimators']
            self.X_era = copx



        #    self.X_v = copxv
            # apply result of hyperopt hyperparameter tuning
          # lg_fu = xgbr(**initg)

            #model_g.set_params(**optparams)

            #model_g.fit(self.X_era,self.y_t, eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)])
      #     ft_name = self.X_era.columns

        #ft_imp= model_g.feature_importances_


        #sort_imp_ind = np.argsort(-ft_imp)

        #sorted_ft_name  = self.X_era.columns[sort_imp_ind]

        #ft_name1= sorted_ft_name[0:int(perc * len(sorted_ft_name))]

        #self.drop_feat_imp1 = self.X_era.drop(ft_name1, axis=1 ).columns

        self.drop_feat_imp1 = self.PermutationImp(model_g)

        drop_feat= pd.DataFrame(index = np.arange(0,self.drop_feat_imp1.shape[0],1))

        drop_feat['drop_feat1'] = self.drop_feat_imp1

        drop_feat.to_csv('/NumeraiGCPreal/MDAdrop_feat'+str(self.ms)+'.csv',encoding='utf-8', index=False)



     #   drop_feat= pd.DataFrame()
    #        drop_feat2= pd.DataFrame()
     #   drop_feat['drop_feat1']= self.drop_feat_imp1
    #       drop_feat2['drop_feat2']= self.drop_feat_imp2
     #   drop_feat.to_csv('/content/gdrive/My Drive/f0/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv',encoding='utf-8', index=False)

        return self
      #      drop_feat2.to_csv('/content/drive/My Drive/drop_feat2.csv',encoding='utf-8', index=False)
            #plot_importance(model_g,fig_size=(160,100),color=colorplot)



     #Return Pearson product-moment correlation coefficients.
    @staticmethod
    def CorrelationScore(y_true,y_pred) -> np.float64:
        frame=pd.DataFrame()
        frame['true']=y_true
        frame['pred']=y_pred
        return np.corrcoef(frame['true'],frame['pred'])[0,1]



    @staticmethod
    # Numerai's primary scoring metric
    def numerai_corr(preds, target)-> np.float64:
        # rank (keeping ties) then gaussianize predictions to standardize prediction distributions
        ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
        gauss_ranked_preds = stats.norm.ppf(ranked_preds)
        # center targets around 0
        centered_target = target - target.mean()
        # raise both preds and target to the power of 1.5 to accentuate the tails
        preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
        target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
        # finally return the Pearson correlation
        return np.corrcoef(preds_p15, target_p15)[0, 1]

    @staticmethod
    def SaveModel(model,model_file_name,t,path=None):
        if t =='c':
            model.save_model(path+model_file_name)
        else:
            pickle.dump(model,open(path+model_file_name+'.pkl',"wb"))

    @staticmethod
    def LoadModel(model_file_name,t,path=None)->object:
        if t =='c':
            model = CatBoostRegressor().load_model(path+model_file_name)
        else:
            model = pickle.load(open(path+model_file_name+'.pkl',"rb"))
        return model

    #@staticmethod
    #def SilhouetteScore(estimator,X):
    #    clusters = estimator.fit_predict(X)

    #    score = sc(X,clusters,metric = 'euclidean')

    #    return score


    def FitEnsembleOverEra(self,gridg,Npochs= 1 , sub_splits = 3 , FeatImpSelection = False):

        st_time = time.time()

        initg = {
        "n_estimators" :3,
        "max_depth" : 7,
              "learning_rate" : 0.01,
              "eval_metric":"rmse"
          #    "early_stopping_rounds": 2,
        #      "feature_selector":'greedy'
   #     "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
                    }

        if FeatImpSelection:

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)

            self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)
        # overrride indexable setting parameter from dataset era
        #

        ptg =  PurgedTimeSeriesSplitGroups(groups = self.X_era[self.cat_col], n_splits=sub_splits)

        for n in range(Npochs):



            model_g = xgbr(**initg)


            bscv = GridSearchCV(model_g,gridg, cv = ptg)

            bscv.fit(self.X_era,self.y_t)


            optparams = bscv.best_params_
#
            model_g.set_params(**optparams)

            model_g.fit(self.X_era,self.y_t,eval_set = [(self.X_era,self.y_t),(self.X_v,self.y_v)])

            print("for XGBoostRegressor @ {} Fold/Model the Training Corr is {}".format(self.ms,
                              self.CorrelationScore(self.y_t,model_g.predict(self.X_era))))


            print("for XGBoostRegressor @ {} Fold/Model the Validation Corr is {}".format(self.ms,
                              self.CorrelationScore(self.y_v,model_g.predict(self.X_v))))



        self.SaveModel(model_g,'XGBoostRegressor Model'+' Model_number:'+str(self.ms),'g',path='/NumeraiGCPreal/numeraimodels/')

        loss_curvey = model_g.evals_result()['validation_0']['rmse']

        loss_curveyv = model_g.evals_result()['validation_1']['rmse']

        fig, ax = plt.subplots(2,1)

        ax[0].plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')

        ax[1].plot(np.arange(0, len(loss_curveyv)),loss_curveyv,c='r',label = ' Validation loss RMSE over CV iter')

        ax[0].set_xlabel('Iterations')
        ax[1].set_xlabel('Iterations')


        ax[0].set_ylabel('RMSE loss curve')
        ax[1].set_ylabel('RMSE loss curve')


        ax[0].legend()
        ax[1].legend()
 #
        plt.show()

        st_time2= time.time()

        print('Time taken to tune XGBR model wi th GridSearchCV {}'.format(st_time2 - st_time))

    def RollingFitEnsembleOverEra(self,Npochs= 1 , sub_splits = 3 , FeatImpSelection = True):

        st_time = time.time()

        initg = {

          #  "n_estimators" :125,
            "max_depth" : 8,
            #"learning_rate" : 0.01,
            "eval_metric":"rmse",
         #   "warm_start":True
            #"early_stopping_rounds": 2,
        #    "feature_selector":'greedy'
       #     "colsample_bytree" : 0.1,
       #     "tree_method" : 'gpu_hist'

        }

        gridg = {

         #  'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
            'eta': hp.quniform('eta', 0.001, 0.09, 0.001),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            #'max_depth':  hp.choice('max_depth', np.arange(4, 7, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 10, 1)),
        #   'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 3, 0.5),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.8, 1, 0.05),
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
        #   'nthread': 4,
          #  'booster': 'gbtree',
        #   'tree_method': 'exact',
        #   'silent': 1,
        #
        }

        frwindow = 1
        spread = 0
        self.X_era['eraB'] = np.arange(0,self.X_era.shape[0],1)
        copx = self.X_era.copy()
        #copxv= self.X_v.copy()
        mse= 0
        frwindow = 1
        ue= self.X_era['era'].unique()
        #uv = self.X_v['era'].unique()
        #erawindows = np.intersect1d(ue,uv)
        erawindows = ue
        targ = [ c for c in self.X_era.columns if c.startswith('target')]



        if FeatImpSelection:

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            #self.drop_feat_imp = [f for f in self.drop_feat_imp1 if f not in targ]

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)


#            self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)
        # overrride indexable setting parameter from dataset era
        #ptg =  PurgedTimeSeriesSplitGroups(groups = self.X_era[self.cat_col], n_splits=sub_splits)
        def RollingOptimizeMSE(params):

            model_m = xgbr(**initg)
            model_m.set_params(**params)
            mse = 0
                # rolling sliding winodow over era

            for col, er in enumerate(erawindows):


                self.X_era = self.X_era[ self.X_era['era'] == er ]

                self.y_t = self.X_era['target']

                self.X_era= self.X_era.drop(targ,axis =1)




                ptg = PurgedTimeSeriesSplitGroups(groups = self.X_era['eraB'])

                for tn, (trainingrow, testingrow) in enumerate(ptg.split(self.X_era,y= self.y_t)):

                    #self.X_v = self.X_v.drop(targ,axis = 1)
                    #self.X_era = pd.concat([self.X_era.iloc[roll:,:],self.X_v.iloc[0:roll,:]],axis=0)
                    #self.y_t= pd.concat([self.y_t[roll:],self.y_v[0:roll]],axis = 0 )
                    # consider the last split over the era

                    if tn==4:


                        model_m.fit(self.X_era.drop(['era','eraB'],axis=1).iloc[trainingrow,:].values,self.y_t.values[trainingrow].reshape(-1,1).ravel())

                        predth = model_m.predict(self.X_era.drop(['era','eraB'],axis=1).iloc[testingrow,:].values)

                        mse += mean_squared_error(self.y_t.values[testingrow].reshape(-1,1), predth)

                        #model_m.n_estimators += initg['n_estimators']

                        self.X_era= copx


                    else:

                        pass



         #       self.X_v = copxv


            lossh = mse
            return {'loss':lossh,'status':STATUS_OK}

        trials = Trials()
        optparamsbayes = fmin(RollingOptimizeMSE,space = gridg,algo=tpe.suggest,max_evals=self.max_eval,trials=trials)
        optparams = optparamsbayes

        ue= self.X_era['era'].unique()
        #uv = self.X_v['era'].unique()
        erawindows = ue
        #erawindows = np.intersect1d(ue,uv)


        model_g = xgbr(**initg)
        model_g.set_params(**optparams)



        for er in erawindows:


            self.X_era = self.X_era[ self.X_era['era'] == er ]

           # self.X_v = self.X_v[self.X_v['era']== er]

            self.y_t = self.X_era['target']

          #  self.y_v = self.X_v['target']

            self.X_era= self.X_era.drop(targ,axis =1)

            cols = self.X_era.columns

          #  self.X_v= self.X_v[cols]

         #   self.X_v = self.X_v.drop(targ,axis = 1)

            # bscv = GridSearchCV(model_g,gridg, cv = ptg)

            #    bscv.fit(self.X_era,self.y_t)


            #    optparams = bscv.best_params_


            model_g.fit(self.X_era.drop(['era','eraB'],axis=1),self.y_t,eval_set = [(self.X_era.drop(['era','eraB'],axis=1),self.y_t)])



            print("for XGBoostRegressor @ {} Fold/Model the Training Corr is {}".format(self.ms,self.CorrelationScore(self.y_t,model_g.predict(self.X_era.drop(['era','eraB'],axis=1)))))

           # model_g.n_estimators += initg['n_estimators']


         #   print("for XGBoostRegressor @ {} Fold/Model the Validation Corr is {}".format(self.ms,self.CorrelationScore(self.y_v,model_g.predict(self.X_v))))

            self.X_era = copx

           # self.X_v = copxv
                #end loop

        self.SaveModel(model_g,'XGBoostRegressor Model'+' Model_number:'+str(self.ms),'g',path='/NumeraiGCPreal/numeraimodels/')

        loss_curvey = model_g.evals_result()['validation_0']['rmse']

     #   loss_curveyv = model_g.evals_result()['validation_1']['rmse']

        fig, ax = plt.subplots()

        ax.plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')

    #    ax[1].plot(np.arange(0, len(loss_curveyv)),loss_curveyv,c='r',label = ' Validation loss RMSE over CV iter')

        ax.set_xlabel('Iterations')
   #     ax[1].set_xlabel('Iterations')


        ax.set_ylabel('RMSE loss curve')
  #      ax[1].set_ylabel('RMSE loss curve')


        ax.legend()
   #     ax[1].legend()
#
        plt.show()

        st_time2= time.time()

        print('Time taken to tune XGBR model wi th GridSearchCV {}'.format(st_time2 - st_time))

    #avergage over era model
    def BaggedEnsembleRollingFitEnsembleOverEra(self, Npochs= 1, sub_splits = 3, FeatImpSelection = False, debug = False):

        st_time = time.time()

        initg = {

          #  "n_estimators" :125,
            "max_depth" : 8,
            #"learning_rate" : 0.01,
            "eval_metric":"rmse",
         #   "warm_start":True
            #"early_stopping_rounds": 2,
        #    "feature_selector":'greedy'
       #     "colsample_bytree" : 0.1,
       #     "tree_method" : 'gpu_hist'

        }

        gridg = {

        #   'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
            'eta': hp.quniform('eta', 0.001, 0.09, 0.001),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            #'max_depth':  hp.choice('max_depth', np.arange(4, 7, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 10, 1)),
        #   'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 3, 0.5),
           # 'colsample_bytree': hp.quniform('colsample_bytree', 0.8, 1, 0.05),
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
        #   'nthread': 4,
          #  'booster': 'gbtree',
        #   'tree_method': 'exact',
        #   'silent': 1,
        #
        }

        frwindow = 1
        spread = 0
        self.X_era['eraB'] = np.arange(0,self.X_era.shape[0],1)
        copx = self.X_era.copy()
#        copxv= self.X_v.copy()
        mse= 0
        frwindow = 1
        ue= self.X_era['era'].unique()
       # uv = self.X_v['era'].unique()
        #erawindows = np.intersect1d(ue,uv)
        erawindows = ue
        targ = [ c for c in self.X_era.columns if c.startswith('target')]
        model_gs = []
        ws= []
        fig, ax = plt.subplots(len(ue), 1)



        if FeatImpSelection:

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            #self.drop_feat_imp = [f for f in self.drop_feat_imp1 if f not in targ]

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)


#            self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)
        # overrride indexable setting parameter from dataset era
        #ptg =  PurgedTimeSeriesSplitGroups(groups = self.X_era[self.cat_col], n_splits=sub_splits)\

        if debug:
            try:
              self.y_t = self.X_era['target']

              self.X_era= self.X_era.drop(targ,axis =1)
              for idx, e in enumerate(ue):

                  model_g = self.LoadModel(f'XGBoostRegressor Model Model_number:{e}{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')

                  ws = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+'weights'+'.csv')["MSEs"]

                  print(ws)

                  ws = ws.values

                  print(ws[idx])


                  if idx ==0:
                      predg = (ws[idx] * model_g.predict(self.X_era.drop(['era','eraB'],axis= 1)))/len(ue)

                  else:
                      print('shape of predg {}'.format(predg.shape))
                      print('shape of ws[idx] {}'.format(ws.shape))
                      print('shape of next pred {}'.format(model_g.predict(self.X_era.drop(['era','eraB'],axis= 1)).shape))
                      predg += (ws[idx] * model_g.predict(self.X_era.drop(['era','eraB'],axis= 1)))/(len(ue))

                  print(" @ era {} for XGBoostRegressor sub ensemble w lag {} and corr score {}".format(e, self.ms,self.CorrelationScore(self.y_t,predg)))
                  print(predg)



            except:
                raise ValueError('Ensemble Vectorization Failed')

        if not debug:

            for idx, e in enumerate(ue):

                def RollingOptimizeMSE(params):

                    model_m = xgbr(**initg)
                    model_m.set_params(**params)
                    mse = 0
                        # rolling sliding winodow over era

                    self.X_era = self.X_era[ self.X_era['era'] == e ]

                    self.y_t = self.X_era['target']

                    self.X_era= self.X_era.drop(targ,axis =1)

                    ptg = PurgedTimeSeriesSplitGroups(groups = self.X_era['eraB'])

                    for tn, (trainingrow, testingrow) in enumerate(ptg.split(self.X_era,y= self.y_t)):

                            #self.X_v = self.X_v.drop(targ,axis = 1)
                            #self.X_era = pd.concat([self.X_era.iloc[roll:,:],self.X_v.iloc[0:roll,:]],axis=0)
                            #self.y_t= pd.concat([self.y_t[roll:],self.y_v[0:roll]],axis = 0 )
                            # consider the last split over the era

                        if tn==4:

                            model_m.fit(self.X_era.drop(['era','eraB'],axis=1).iloc[trainingrow,:].values,self.y_t.values[trainingrow].reshape(-1,1).ravel())

                            predth = model_m.predict(self.X_era.drop(['era','eraB'],axis=1).iloc[testingrow,:].values)

                            mse += mean_squared_error(self.y_t.values[testingrow].reshape(-1,1), predth)

                            self.X_era= copx

                        else:

                            pass

                    lossh = mse
                    return {'loss':lossh,'status':STATUS_OK}

                trials = Trials()
                optparamsbayes = fmin(RollingOptimizeMSE,space = gridg,algo=tpe.suggest,max_evals=self.max_eval,trials=trials)
                optparams = optparamsbayes
                #erawindows = np.intersect1d(ue,uv)

                self.X_era = self.X_era[ self.X_era['era'] == e ]

                self.y_t = self.X_era['target']

                self.X_era= self.X_era.drop(targ,axis =1)

                model_g = xgbr(**initg)
                model_g.set_params(**optparams)
                model_g.fit(self.X_era.drop(['era','eraB'],axis =1 ),self.y_t)
                predgt = model_g.predict(self.X_era.drop(['era','eraB'],axis = 1))
                mses = mean_squared_error(self.y_t, predgt)

                ws.append(mses)
                self.SaveModel(model_g,f'XGBoostRegressor Model Model_number:{e}{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')
                model_gs.append(model_g)
                self.X_era = copx
            #end loop

        # vectorize list
            ws = np.array(ws)
            ws = ws.reshape(-1,)
            ws = ws/np.sum(ws)

            weightframe = pd.DataFrame()
            weightframe["MSEs"] = ws


            print('scaled weightframe successs')
            print(weightframe)
            weightframe.to_csv('/NumeraiGCPreal/CURRDATA/'+'weights'+'.csv',index=False)

            model_gs = np.array(model_gs)
            model_gs = model_gs.reshape(-1,)
            #predg = np.zeros(self.X_era.shape[0])
            self.y_t = self.X_era['target']

            self.X_era= self.X_era.drop(targ,axis =1)
            mwp = list(zip(model_gs, ws))
            model_EWM= WeightedSumModel(mwp)
            self.SaveModel(model_EWM,f'XGBoostRegressor Model Model_number: Bagged ensemble','g',path='/NumeraiGCPreal/numeraimodels/')
            # loop over era ensemble weighted model
            for (model, (idx, er)) in zip(model_gs,enumerate(ue)):


                if idx ==0:
                      predg = (ws[idx] * model.predict(self.X_era.drop(['era','eraB'],axis= 1)))/len(ue)

                else:
                      print('shape of predg {}'.format(predg.shape))
                      print('shape of ws[idx] {}'.format(ws.shape))
                      print('shape of next pred {}'.format(model.predict(self.X_era.drop(['era','eraB'],axis= 1)).shape))
                      predg += (ws[idx] * model.predict(self.X_era.drop(['era','eraB'],axis= 1)))/(len(ue))

                print(" @ era {} for XGBoostRegressor sub ensemble w lag {} and corr score {}".format(er, self.ms,self.CorrelationScore(self.y_t,predg)))


            predg_ewm = model_EWM.predict(self.X_era.drop(['era','eraB'],axis= 1))



                #loss_curvey = model_g.evals_result()['validation_0']['rmse']
                #ax[idx].plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')
                #ax[idx].set_xlabel('Iterations')
               # ax[idx].set_ylabel('RMSE loss curve')
               # ax[idx].legend()

        print(" TOTAL ensemble over {} eras for XGBoostRegressor w lag {} and corr score for EWM {}".format(er, self.ms,self.CorrelationScore(self.y_t,predg_ewm)))

        print(" TOTAL ensemble over {} eras for XGBoostRegressor w lag {} and corr score {}".format(ue, self.ms,self.CorrelationScore(self.y_t,predg)))
#
        plt.show()

        st_time2= time.time()

        print('Time taken to tune XGBR model wi th GridSearchCV {}'.format(st_time2 - st_time))




    def BooostedEnsembleRollingFitEnsembleOverEra(self,Npochs= 1 , sub_splits = 3 , FeatImpSelection = False,debug= False):

        st_time = time.time()

        initg = {

          #  "n_estimators" :125,
            "max_depth" : 8,
            #"learning_rate" : 0.01,
            "eval_metric":"rmse",
         #   "warm_start":True
            #"early_stopping_rounds": 2,
        #    "feature_selector":'greedy'
       #     "colsample_bytree" : 0.1,
       #     "tree_method" : 'gpu_hist'

        }

        gridg = {

        #   'n_estimators': hp.choice('n_estimators', range(100, 1000, 100)),
            'eta': hp.quniform('eta', 0.001, 0.09, 0.001),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            #'max_depth':  hp.choice('max_depth', np.arange(4, 7, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 10, 1)),
        #   'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 3, 0.5),
             'lambda': hp.quniform('lambda', 0.5, 3, 0.5),
           # 'colsample_bytree': hp.quniform('colsample_bytree', 0.8, 1, 0.05),
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
        #   'nthread': 4,
          #  'booster': 'gbtree',
        #   'tree_method': 'exact',
        #   'silent': 1,
        #
        }

        frwindow = 1
        spread = 0
        self.X_era['eraB'] = np.arange(0,self.X_era.shape[0],1)
        #copxv= self.X_v.copy()
        mse= 0
        frwindow = 1
        ue= self.X_era['era'].unique()
        #uv = self.X_v['era'].unique()
        #erawindows = np.intersect1d(ue,uv)
        erawindows = ue
        targ = [ c for c in self.X_era.columns if c.startswith('target')]
        model_gs = []
        ws= []
        metafeaturestrain = np.zeros((self.X_era.shape[0], len(ue)))
        #metafeaturestest = np.zeros((self.X_era.shape[0], len(ue)))
        copx = self.X_era.copy()


        fig, ax = plt.subplots()

        if FeatImpSelection:

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            #self.drop_feat_imp = [f for f in self.drop_feat_imp1 if f not in targ]

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)
#            self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)
        # overrride indexable setting parameter from dataset era
        #ptg =  PurgedTimeSeriesSplitGroups(groups = self.X_era[self.cat_col], n_splits=sub_splits


        if debug:
            self.y_t = self.X_era['target']
            self.X_era= self.X_era.drop(targ,axis =1)

            for idx, e in enumerate(ue):

                model_g = self.LoadModel(f'BoostedXGBoostRegressor Model Model_number:{e}{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')
                model_gs.append(model_g)
                metafeaturestrain[:,idx] = model_g.predict(self.X_era.drop(['era','eraB'],axis = 1))



            model_gs = np.array(model_gs)
            model_gs = model_gs.reshape(-1,)
            #predg = np.zeros(self.X_era.shape[0])
            # loop over era ensemble weighted model
            modelboostg = xgbr(**initg)
            #subsume the last eras params
            optparams = model_g.get_params()
            modelboostg = modelboostg.set_params(**optparams)
            modelboostg.fit(metafeaturestrain, self.y_t)
            predg = modelboostg.predict(metafeaturestrain)
            print(" @ era for XGBoostRegressor sub ensemble w lag {} and corr score {}".format(self.ms,self.CorrelationScore(self.y_t,predg)))
            loss_curvey = modelboostg.evals_result()['validation_0']['rmse']
            self.SaveModel(modelboostg,f'MXGBoostRegressor Model Model_number:{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')
            ax.plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('RMSE loss curve')
            ax.legend()
            print(" TOTAL ensemble over {} eras for XGBoostRegressor w lag {} and corr score {}".format(len(ue), self.ms,self.CorrelationScore(self.y_t,predg)))
            plt.show()

            st_time2= time.time()
            print('Time taken to tune XGBR model with GridSearchCV {}'.format(st_time2 - st_time))


        else:

            for idx, e in enumerate(ue):

                def RollingOptimizeMSE(params):

                    model_m = xgbr(**initg)
                    model_m.set_params(**params)
                    mse = 0
                        # rolling sliding winodow over era

                    self.X_era = self.X_era[ self.X_era['era'] == e ]

                    self.y_t = self.X_era['target']

                    self.X_era= self.X_era.drop(targ,axis =1)

                    ptg = PurgedTimeSeriesSplitGroups(groups = self.X_era['eraB'])

                    for tn, (trainingrow, testingrow) in enumerate(ptg.split(self.X_era,y= self.y_t)):

                            #self.X_v = self.X_v.drop(targ,axis = 1)
                            #self.X_era = pd.concat([self.X_era.iloc[roll:,:],self.X_v.iloc[0:roll,:]],axis=0)
                            #self.y_t= pd.concat([self.y_t[roll:],self.y_v[0:roll]],axis = 0 )
                            # consider the last split over the era

                        if tn==4:

                            model_m.fit(self.X_era.drop(['era','eraB'],axis=1).iloc[trainingrow,:].values,self.y_t.values[trainingrow].reshape(-1,1).ravel())

                            predth = model_m.predict(self.X_era.drop(['era','eraB'],axis=1).iloc[testingrow,:].values)

                            mse += mean_squared_error(self.y_t.values[testingrow].reshape(-1,1), predth)

                            self.X_era= copx

                        else:
                            pass

                    lossh = mse
                    return {'loss':lossh,'status':STATUS_OK}

                trials = Trials()
                optparamsbayes = fmin(RollingOptimizeMSE,space = gridg,algo=tpe.suggest,max_evals=self.max_eval,trials=trials)
                optparams = optparamsbayes

                self.y_t = self.X_era['target']

                self.X_era= self.X_era.drop(targ,axis =1)
                model_g = xgbr(**initg)
                model_g.set_params(**optparams)
                model_g.fit(self.X_era.drop(['era','eraB'],axis =1 ),self.y_t)
                print('Shape of pred {}'.format(model_g.predict(self.X_era.drop(['era','eraB'],axis = 1)).shape))
                metafeaturestrain[:,idx] = model_g.predict(self.X_era.drop(['era','eraB'],axis = 1))

                self.SaveModel(model_g,f'BoostedXGBoostRegressor Model Model_number:{e}{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')
                model_gs.append(model_g)
                self.X_era = copx
            #end loop

          # vectorize list
            model_gs = np.array(model_gs)
            model_gs = model_gs.reshape(-1,)
            #predg = np.zeros(self.X_era.shape[0])
            self.y_t = self.X_era['target']
            self.X_era= self.X_era.drop(targ,axis =1)
            # loop over era ensemble weighted model
            modelboostg = xgbr(**initg)
            #subsume the last eras params
            modelboostg = modelboostg.set_params(**optparams)
            modelboostg.fit(metafeaturestrain, self.y_t)
            predg = modelboostg.predict(metafeaturestrain)
            print(" @ era for XGBoostRegressor sub ensemble w lag {} and corr score {}".format(self.ms,self.CorrelationScore(self.y_t,predg)))
            loss_curvey = modelboostg.evals_result()['validation_0']['rmse']
            self.SaveModel(modelboostg,f'MXGBoostRegressor Model Model_number:{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')
            ax.plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('RMSE loss curve')
            ax.legend()
            print(" TOTAL ensemble over {} eras for XGBoostRegressor w lag {} and corr score {}".format(len(ue), self.ms,self.CorrelationScore(self.y_t,predg)))
            plt.show()
            st_time2= time.time()
            print('Time taken to tune XGBR model wi th GridSearchCV {}'.format(st_time2 - st_time))


    #def ClusterFeatures(self) -> pd.DataFrame():

    #    model_gm = GMM(init_params ='kmeans', random_state=0)

    #    cv =[(slice(None),slice(None))]

    #    grid_search = {'n_components':np.arange(2,8,1),'tol':np.arange(0.0001,0.0005,0.0001)}

    #    cluster_gsv = GridSearchCV(model_gm,grid_search,n_jobs = -1, scoring = self.SilhouetteScore)

    #    cluster_gsv.fit(self.X_era.T)


    #    model_gm.set_params(**cluster_gsv.best_params_)

    #    print("silhouette score "+ str(self.silhouette_score(model_gm,self.X_era.T)))

    #    cluster_predictions = model_gm.predict(self.X_era.T)


    #    results = pd.DataFrame(  col = 'Clusters' ,index = self.X_era.columns)
    #    results = cluster_predictions
    #    fig , ax = plt.subplot(figsize = (50,40))
    #    results.plot(kind = 'hist', ax = ax, c = 'r',density= True)
    #    ax.set_label('Hisogram')
    #    ax.set_title('Histogram of Clusters of Features')
    #    ax.set_xlabel('Cluster #')
    #    ax.set_ylabel('Frequency')
    #    ax.legend()
    #    plt.show()


    #    return results


    def AggregatePoorEras(self,FeatImpSelection=True,bins=50) -> pd.DataFrame():
        st_time = time.time()
        MSEm =[]

        if FeatImpSelection:

            #self.ms = m

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)

           # self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)


        model_g = self.LoadModel('XGBoostRegressor Model'+' Model_number:'+str(self.ms),'g',path='/NumeraiGCPreal/numeraimodels/')

        aera = np.unique(self.X_era[self.cat_col].values.reshape( -1,1))
        cnt = 0

        for e in aera:

            print(e)

            self.X_era1 = self.X_era.loc[self.X_era[self.cat_col]==e]


            localy_t = self.X_era1['target'].values.reshape(-1,1)
            # do this once

            targ = [ c for c in self.X_era1.columns if c.startswith('target')]


          #  self.X_newera= pd.concat([self.X_era,self.X_v]).sort_values(by='era',ascending =True)
         #   self.y_newt= self.X_newera['target']
            self.X_era1= self.X_era1.drop(targ,axis =1)


            self.X_era1 = self.X_era1.drop('id',axis =1 )

            targ = []
            # agg over era

            p = model_g.predict(self.X_era1)
            # calculate mse over era slice

            MSEm.append( np.sum(np.sqrt((localy_t - p)**2 / len(localy_t))))

            cnt += 1

        MSEm = np.array(MSEm,dtype='float64').reshape( -1,1)
        print('The shape of MSEm is {}'.format(MSEm.shape))
        threshold = np.mean(MSEm) + (np.std(MSEm))*2

        ind=np.where(MSEm > threshold)
        print(threshold)
       # print(ind)
        #try:
        #    MSEmo = MSEm[ind]

       #     aaera = aera[ind]
      #      print('worked')
     #   except:
      #        pass


        aggera = pd.DataFrame(MSEm, columns = ['EraP'])
        aggera['Era'] = aera

        aggera.index = np.arange(0 , len(MSEm),1)
        aggera = aggera.sort_values(by='EraP', ascending = False)
        aggera.to_csv('/NumeraiGCPreal/CURRDATA/'+str(self.ms)+'aggera.csv')
        aggera.head()

        fig, ax = plt.subplots(figsize=(8,6))

        ax.hist(MSEm, density=True, bins=bins, alpha=0.1, color='r',label='MSE Era Hist')
        sns.kdeplot(data=MSEm, color ='b',ax=ax, label ='KDE curve of MSE Era hist')

# consider after stat threhold set

    #    ax[1].hist(MSEmo, density=True, bins=bins, alpha=0.1, c = 'b')
    #    sns.kdeplot(data=MSEmo, x="Kde", ax=ax[1])



        plt.show()

        return aggera


    def EnsembleOverPoorEra(self,gridg,FeatImpSelection = False, sub_splits= 3, Npochs =1) :

        st_time = time.time()
        cnt =0

        if FeatImpSelection:

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)

            self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)

            self.X_era = self.X_era.drop('id',axis =1 )

            self.X_v = self.X_v.drop('id',axis =1)

        # overrride indexable setting parameter from dataset era
        #
      # the 2/3 of the worst performing eras
      # taking too little eras will underfit the model
      #
        agg = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+str(self.ms)+'aggera.csv')['Era'][0:44]
        agg = agg.sort_values(ascending = True)

        # debug the isin era selection
    #    print(np.unique(agg.values.reshape( -1,1)))
    #    print(np.unique(self.X_era[self.cat_col].values.reshape(-1,1)))
     #   print(self.X_era[self.cat_col].isin(agg.values.reshape(-1,1).T))
     #  dim(agg) -> (len(v),)
        if cnt ==0:

          #  print(self.X_era[self.cat_col].values)
            #print(self.X_era.duplicated().shape)
            #print(self.X_era.shape)
            #print(type(agg))

            self.X_era.head()
            self.X_era = self.X_era[self.X_era[self.cat_col].isin(agg.values)]
            # RAM conservation
        #    del [[self.X_era1]]
            #self.X_era1 = []
          #  self.X_newera= pd.concat([self.X_era,self.X_v]).sort_values(by='era',ascending =True)
         #   self.y_newt= self.X_newera['target']
            ptg =  PurgedTimeSeriesSplitGroups(groups = self.X_era[self.cat_col], n_splits=sub_splits)
            self.y_t = self.X_era['target']
            self.y_v = self.X_v['target']
            targ = [ c for c in self.X_era.columns if c.startswith('target')]

            #self.X_era= self.X_era.drop(targ,axis =1)
           # self.X_v= self.X_v.drop(targ,axis =1)
            self.X_era= self.X_era.drop('target',axis =1)
            self.X_v= self.X_v.drop('target',axis =1)

#            self.X_era = self.X_era.drop('id',axis =1 )
#            self.X_v= self.X_v.drop('id',axis=1)

            # make sure it is in sequential order
            self.X_era = self.X_era.sort_values(by = self.cat_col,ascending = True)

            targ = []

            initg = {
        "n_estimators" : 150,
       # "max_depth" : 5,
              "learning_rate" : 0.01,
              "eval_metric":"rmse"
          #    "early_stopping_rounds": 2,
        #      "feature_selector":'greedy'
   #     "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
                    }

         #   del [[self.X_era]]
         #   self.X_era=[]

        for n in range(Npochs):



     #   sub_tss= KFold(n_splits = 5 )

            model_g  = xgbr(**initg)


            bscv = GridSearchCV(model_g ,gridg,cv = ptg)

            bscv.fit(self.X_era,self.y_t)

          #  def score(gridg):

           #     model = xgbr(**initg)22
          #      model.fit(self.X_era,self.y_t,eval_set=[(self.X_era, self.y_t),(self.X_v,self.y_v)]
           #     ,verbose=False)

            #    Y_pred = model.predict(self.X_v)
          #      score = np.sqrt(mean_squared_error(self.y_v, Y_pred))
            #print(score)
         #       return {'loss': score, 'status': STATUS_OK}

        #    def opt(trials,gridg):
           #
           #     best = fmin(score,gridg,algo = tpe.suggest, max_evals = 20)

           #     return best


           # trials = Trials()

          #  optparams = opt(trials,gridg)

            optparams = bscv.best_params_

        # apply result of hyperopt hyperparameter tuning


            model_g.set_params(**optparams)

            model_g.fit(self.X_era,self.y_t,eval_set = [(self.X_era,self.y_t),(self.X_v,self.y_v)])

            print("for XGBoostRegressor @ {} Fold/Model the Training Corr is {}".format(self.ms,
                              self.CorrelationScore(self.y_t,model_g.predict(self.X_era))))


            print("for XGBoostRegressor @ {} Fold/Model the Validation Corr is {}".format(self.ms,
                              self.CorrelationScore(self.y_v,model_g.predict(self.X_v))))
     #


        loss_curvey = model_g.evals_result()['validation_0']['rmse']

        loss_curveyv = model_g.evals_result()['validation_1']['rmse']

        fig, ax = plt.subplots(2,1)

        ax[0].plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')

        ax[1].plot(np.arange(0, len(loss_curveyv)),loss_curveyv,c='r',label = ' Validation loss RMSE over CV iter')

        ax[0].set_xlabel('Iterations')
        ax[1].set_xlabel('Iterations')


        ax[0].set_ylabel('RMSE loss curve ')
        ax[1].set_ylabel('RMSE loss curve')


        ax[0].legend()
        ax[1].legend()
 #
        plt.show()

        self.SaveModel(model_g,'PoorEraXGBoostRegressor Model'+' Model_number:'+str(self.ms),'g',path='/NumeraiGCPreal/numeraimodels/')

        st_time2= time.time()

        print('Time taken to tune XGBR model for poor eras with GridSearchCV {}'.format(st_time2 - st_time))


    def AggregatePoorTargets(self,FeatImpSelection = False,bins = 50)-> pd.DataFrame():

        st_time = time.time()
        MSEm =[]

        if FeatImpSelection:

            #self.ms = m

            self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(self.ms)+'.csv')['drop_feat1'].values

            self.X_era= self.X_era.drop(self.drop_feat_imp1,axis = 1)

            self.X_era = self.X_era.drop('id',axis =1 )

           # self.X_v = self.X_v.drop(self.drop_feat_imp1,axis = 1)


        model_g = self.LoadModel('XGBoostRegressor Model'+' Model_number:'+str(self.ms),'g',path='/NumeraiGCPreal/numeraimodels/')

        targ = [ c for c in self.X_era.columns if c.startswith('target')]

        print("Number of targ {}".format(len(targ)))


        for t in targ:

            MSEm.append(np.sum(np.sqrt((self.X_era[t].values.reshape( -1,1) - model_g.predict(self.X_era.drop(targ,axis =1).values).reshape(-1,1))**2 / self.X_era.shape[0])))

        targetper = pd.DataFrame(MSEm, columns = ['Target_MSE'])
        targetper.index = targ
        targetper = targetper.sort_values(by= 'Target_MSE',ascending = True)

        targetper.to_csv('/NumeraiGCPreal/CURRDATA/'+str(self.ms)+'aggtarget.csv')

        fig, ax = plt.subplots(figsize=(8,6))

        ax.hist(MSEm, density=True, bins=bins, alpha=0.1, color='r',label='MSE Target Hist')
        sns.kdeplot(data=MSEm, color ='b',ax=ax, label ='KDE curve of MSE target hist')

        plt.show()


    def PredictSubmit(self,model_id,n_splits = 3,FeatImpSelection= True):

        pred_ct= np.zeros(self.X_live.shape[0])
       # pred_ctp= np.zeros(self.X_live.shape[0])

      #  org = self.X_live.copy()

       # self.X_live.head()
     #   for m in range(1):

        if FeatImpSelection:

              #targ = [ c for c in self.X_era.columns if c.startswith('target')]

            #for m in range(0,self.ms):
              m = self.ms

              self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(m)+'.csv')['drop_feat1'].values

              self.drop_feat_imp = [f for f in self.drop_feat_imp1 if f not in self.targ]



              cols= self.X_era.columns

              global feature_cols, model


             #   self.X_live = org
                # feature selection dim reductio
              #  tree based
              #self.X_live= self.X_live

             # self.X_live = self.X_live[cols]

              #self.X_v= self.X_v[cols]

              model_g = self.LoadModel('XGBoostRegressor Model'+' Model_number:'+str(m),'g',path='/NumeraiGCPreal/numeraimodels/')
             # model_g2 = self.LoadModel('XGBoostRegressor Model'+' Model_number:'+str(m+1),'g',path='/content/gdrive/My Drive/f0/numeraimodels/')
             # model_gp=self.LoadModel('PoorEraXGBoostRegressor Model'+' Model_number:'+str(m),'g',path='/content/gdrive/My Drive/f0/numeraimodels/')
              feature_cols = cols
              subdict ={'enable_categorical':True}
              model= model_g
              self.X_live['era'] = int(self.curr_round -1)
              self.X_live['era'] =  self.X_era['era'].unique().max().astype('int64 ')
              live_features = self.X_live
              #https://colab.research.google.com/github/numerai/example-scripts/blob/master/hello_numerai.ipynb#scrollTo=fSzVtbHsGmkn
              # Wrap your model with a function that takes live features and returns live predictions
              def predict(live_features: pd.DataFrame) -> pd.DataFrame:
                  live_predictions = model.predict(live_features[feature_cols])
                  submission = pd.Series(live_predictions, index=live_features.index)
                  return submission.to_frame("prediction")

              print(predict(live_features))

              p = cloudpickle.dumps(predict)
              predictfunc = "predict_m_"+str(self.ms)+".pkl"
              with open(predictfunc, "wb") as f:
                  f.write(p)
              # Download file if running in Google Colab
              try:

                  files.download(predictfunc)
              except:
                  pass


    def BaggedPredictSubmit(self,model_id,n_splits = 3,FeatImpSelection= True):

        pred_ct= np.zeros(self.X_live.shape[0])

        m = self.ms

#        self.drop_feat_imp1 = pd.read_csv('/content/gdrive/My Drive/f0/CURRDATA/MDAdrop_feat'+str(m)+'.csv')['drop_feat1'].values
#
 #       self.drop_feat_imp = [f for f in self.drop_feat_imp1 if f not in self.targ]

        cols= self.X_era.drop('era',axis=1).columns

        global model_gs, weights, feature_cols


        ue= self.X_era['era'].unique()
        model_gs = []
        for e in range(0,len(ue)):
            model = self.LoadModel(f'XGBoostRegressor Model Model_number:{e}{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')
            model_gs.append(model)

        model_gs = np.array(model_gs)
        model_gs = model_gs.reshape(-1,)
        weights = pd.read_csv('/NumeraiGCPreal/CURRDATA/'+'weights'+'.csv').values.reshape(-1,)
        subdict ={'enable_categorical':True}
        feature_cols = cols
#        model= model_g
        #self.X_live['era'] = int(self.curr_round -1)
        #self.X_live['era'] =  self.X_era['era'].unique().max().astype('int64')

        live_features = self.X_live
        #https://colab.research.google.com/github/numerai/example-scripts/blob/master/hello_numerai.ipynb#scrollTo=fSzVtbHsGmkn
        # Wrap your model with a function that takes live features and returns live predictions
        def predict(live_features: pd.DataFrame) -> pd.DataFrame():
            live_predictions = np.zeros((live_features.shape[0]))
            for (ws, model) in zip(weights,model_gs):
                live_predictions += ws* model.predict(live_features[feature_cols])
            submission = pd.Series(live_predictions, index=live_features.index)
            return submission.to_frame("prediction")

        print(predict(live_features))

        p = cloudpickle.dumps(predict)
        predictfunc = "predict_m_"+str(self.ms)+".pkl"

        with open(predictfunc, "wb") as f:
            f.write(p)
        # Download file if running in Google Colab
        try:

            files.download(predictfunc)
        except:
            pass


    def BoostedPredictSubmit(self,model_id,n_splits = 3,FeatImpSelection= True):

        m = self.ms

        self.drop_feat_imp1 = pd.read_csv('/NumeraiGCPreal/CURRDATA/MDAdrop_feat'+str(m)+'.csv')['drop_feat1'].values

        self.drop_feat_imp = [f for f in self.drop_feat_imp1 if f not in self.targ]

        global feature_cols, model_gs,model_g,ue

        ue= self.X_era['era'].unique()

        cols= self.X_era.drop('era',axis =1).columns
        model_gs = []

        for e in ue:

            model_g = self.LoadModel(f'BoostedXGBoostRegressor Model Model_number:{e}{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/' )
            model_gs.append( model_g)

        model_gs = model_gs.reshape(-1,)
        model_g = self.LoadModel(f'MXGBoostRegressor Model Model_number:{self.ms}','g',path='/NumeraiGCPreal/numeraimodels/')

        feature_cols = cols
        subdict ={'enable_categorical':True}
        model= model_g
        #self.X_live['era'] = int(self.curr_round -1)
        #self.X_live['era'] =  self.X_era['era'].unique().max().astype('int64 ')

        live_features = self.X_live
        #https://colab.research.google.com/github/numerai/example-scripts/blob/master/hello_numerai.ipynb#scrollTo=fSzVtbHsGmkn
        # Wrap your model with a function that takes live features and returns live predictions
        def predict(live_features: pd.DataFrame) -> pd.DataFrame:
            metapredictions = np.zeros((live_features.shape[0],len(ue)))
            for (model_m,(idx, e)) in zip(model_gs,enumerate(ue)):
                metapredictions[:,e] = model_m.predict(live_features[feature_cols])

            live_predictions = model.predict(metapredictions)
            submission = pd.Series(live_predictions, index=live_features.index)
            return submission.to_frame("prediction")

        print(predict(live_features))

        p = cloudpickle.dumps(predict)
        predictfuncname = "boostedpredict_m_"+str(self.ms)+".pkl"

        with open(predictfuncname, "wb") as f:
            f.write(p)
        # Download file if running in Google Colab
        try:

            files.download(predictfuncname)
        except:
            pass

              #pred_ct = model_g.predict(self.X_live)
             # pred_ct2 = model_g.predict(self.X_live)
             # pred_cv = model_g.predict(self.X_v)
             # pred_ctp += model_gp.predict(self.X_live)/float(self.ms)

      #  pred_cta = np.average([pred_ct,pred_ctp],axis=0)

        #results = pd.DataFrame()

        #results['id']=self.t_id

        #results['prediction'] = pred_ct

        #results.to_csv('/content/gdrive/My Drive/f0'+'ROUND'+str(self.curr_round)+'.csv',index = False)



      #  resultsv = pd.DataFrame()

    #    resultsv['id']=self.v_id

    #    resultsv['prediction'] = pred_cv

    #    resultsv.to_csv('/content/gdrive/My Drive/f0'+'VROUND'+str(self.curr_round)+'.csv',index = False)




