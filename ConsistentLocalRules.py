# -*- coding: utf-8 -*-
"""
Algorithms for generating globally consistent rule-based explanations. 

Supplements the paper:  

"Globally-Consistent Rule-Based Summary-Explanations for Machine Learning 
Models: Application to Credit-Risk Evaluation" 

by Cynthia Rudin and Yaron Shaposhnik
        
https://ssrn.com/abstract=3395422

Notation
--------
    n   - number of observations
    p   - number of features
    tau - thresholds 
    x   - observation (length p)
    y_m - prediction by a model (binary)
    X   - Data matrix (n by p matrix)
    Y_global - corresponding labels (length n)
    
        
@author: Yaron Shaposhnik (yaron.shaposhnik@gmail.com)

"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import pulp
from pulp.solvers import GUROBI
import time
import operator
from datetime import datetime
import time

DEBUG = 1
TOL = 10**-6 # tolerance for comparing floating point numbers
TIME_LIMIT = 60 # maximal time to run optimization (in seconds)



def get_time():
    #now = datetime.now()
    #current_time = now.strftime("%H%M%S")
    current_time = str(round(time.time() * 1000))
    return(current_time)


#
# Auxiliary functions - expand data matrixes with complementary values 
#
def complement_binary_series(s):
    s_c = 1-s
    s_c.index = list(map(lambda s_:'~'+s_,s.index.values))
    return(pd.concat([s,s_c]))
    
def complement_continuous_series(s):
    s_c = -s
    s_c.index = list(map(lambda s_:'-'+s_,s.index.values))
    return(pd.concat([s,s_c]))

def assert_cont_expanded(X):
    p = X.shape[1]
    assert(p%2==0), "X is was not continuously expanded, the number of features is odd"
    left = X.iloc[:,:p//2]
    right = X.iloc[:,p//2:]
    left_columns = left.columns.values
    right_columns_adjusted = list(map(lambda s:s[1:],right.columns.values))
    assert((left_columns == right_columns_adjusted).all()), "X is was not continuously expanded, features names do not match"
    right.columns = left.columns.values
    right = -right
    for p_ in range(p//2):
        c = X.columns.values[p_]
        c_left = left.loc[:,c]
        c_right = right.loc[:,c]
        assert((c_left==c_right).all()), 'error with the complement of column ' + str(c)
    #assert(left.equals(right)), "X is was not continuously expanded, values do not match"
        

def complement_binary_dataframe(X):
    X_complement = 1 - X
    X_complement.columns = list(map(lambda s:'~'+s, X_complement.columns.tolist()))
    return(pd.concat([X,X_complement],axis=1))
    
def complement_continuous_dataframe(X):
    X_complement = -X
    X_complement.columns = list(map(lambda s:'-'+s, X_complement.columns.tolist()))
    return(pd.concat([X,X_complement],axis=1))
        


class ConsistentExplanation(BaseEstimator, ClassifierMixin):  
#class ConsistentExplanation():  
    """
    A rule based model. Defined by p thresholds {tau_i}. 
    Predicts True for observations x that satisfy x>=tau and otherwise 
    predicts False. 
    """

    def __init__(self, tau, properties = {}):
        """
        tau : pd.Series, lower bounds of feature names
        properties : dict, optional attributes associated with the model;
                     may include the key 'features' which is a container of 
                     strings holding features names.
        """      
        assert(type(tau)==pd.Series)        
        self.p = len(tau)
        self.tau = tau.copy()
        self.properties = properties        
        assert(self.p>0)
        self.features = tau.index.values.tolist()        
        self.properties['rule'] = self.__str__()
        self.properties['n_terms'] = self.n_terms()
        #self.properties['id'] = hex(id(self))
        #self.properties['all'] = self.__repr__()
        
        
    def __repr__(self): 
        return(str(self.properties))# + str((self.tau,self.ub)))
    
    def to_pd_series(self):
        s = {}
        s['Rule']=self.properties['rule']
        s['Prediction']=self.properties['y_e']
        s['Support']=self.properties['support']
        s['#Features']=self.properties['n_terms']
        s['Runtime']=self.properties['time']
        s['Algorithm']=self.properties['type']       
        #s['x_e']=self.properties['x_e']
        return(pd.Series(s))
        
        
    def __str__(self): 
        '''
        Returns a string representation of the model. 
        Assumes that 
        1) features are of the form 'name', '-name', or '~name' and 
        2)'name' features appear first. 
        Transforms:
        1) -name>=tau_{-name} to name<=tau_{name}, and 
        2) ~name>=tau_{~name} to name<=1-tau_{~name}
        '''       
        skip = []
        predicates = []
        for f in self.features:
            if (f in skip) or np.isinf(self.tau[f]):
                continue
                
            if ('-'+f) in self.features and np.isfinite(self.tau['-'+f]):
                predicates.append('%.2f<=%s<=%.2f'%(self.tau[f],f,-self.tau['-'+f]))
                skip.append('-'+f)
            elif ('~'+f) in self.features and np.isfinite(self.tau['~'+f]):
                predicates.append('%.2f<=%s<=%.2f'%(self.tau[f],f,1-self.tau['~'+f]))
                #if 1-self.tau['~'+f]==self.tau[f]:
                #    predicates.append('%s=%.2f'%(f,self.tau[f]))
                #else:
                #    predicates.append('%.2f<=%s<=%.2f'%(self.tau[f],f,1-self.tau['~'+f]))
                skip.append('~'+f)
            elif f.startswith('-'):
                predicates.append('%s<=%.2f'%(f[1:],-self.tau[f]))
            elif f.startswith('~'):
                predicates.append('%s<=%.2f'%(f[1:],1-self.tau[f]))
            else:
                predicates.append('%s>=%.2f'%(f,self.tau[f]))
        res = ", ".join(predicates) 
        if len(res)==0:
            res='UNDEFINED MODEL'
        return(res)                             


    #def fit(self, X, y=None):
    #    self.classes_ = np.unique([False, True])
    #    return self


    def predict(self, X, y=None):
        tau_sat = (self.tau.values.reshape((1,-1))-TOL<=X).all(axis=1)
        return (np.array(tau_sat).reshape((-1,1)))
       
    
    
    def support(self, X, Y_global, y_m):        
        '''
        Given a data matrix X, computes the number of observations that satsify
        the rule. If these labels are not consistent with y_m return -1.                
        '''       
        Y_rule = self.predict(X).reshape(-1)
        if sum(Y_global[Y_rule] != y_m)>0:
            return(-1)
        else:
            return(sum(Y_rule))

    def set_support(self, X, Y_global, y_m):
        self.properties['support'] = self.support(X, Y_global, y_m)
    
    def set_explained_observation(self, x_e, y_e):
        self.properties['x_e'] = str(x_e.to_dict()).replace(" ","")
        self.properties['y_e'] = y_e

    
    def n_terms(self):        
        '''
        Compute the number of features defining the model (taking into consideration
        features of the form ~name and -name).                
        '''       
        res = 0
        skip = []
        for f in self.features:
            if (f in skip) or np.isinf(self.tau[f]):
                continue
                
            if ('-'+f) in self.features:
                skip.append('-'+f)
            elif ('~'+f) in self.features:
                skip.append('~'+f)
            res+=1
        return(res)                             


    # object comparison functions to check if rules are identical (to remove redundencies)

    def __eq__(self, other):
        if isinstance(other, ConsistentExplanation):
            return ((self.properties['rule'] == other.properties['rule']) and (self.properties['y_e'] == other.properties['y_e']))
        else:
            return False
        
    def __ne__(self, other):
        return (not self.__eq__(other))
    
    def __hash__(self):
        return(hash((self.properties['rule'],self.properties['y_e'])))




def ConsistentExplanation_list_to_df(ce_list):
    l = []
    for ce in ce_list:        
        l.append( ce.to_pd_series())
    res = pd.DataFrame(l)
    return(res)










#
# This is the class the user is going to interact with
#

class ConsistentRulesExplainer():
    """
    An object that generates globally consistent rule-based explanations from data.
    """
                
    def __init__(self, X, Y_global, complement_features=True):    
        '''
        X : pd.DataFrame, data matrix
        Y_global : pd.Series, labels generated by a global model
        '''
        assert(type(X)==pd.DataFrame)
        assert(type(Y_global)==pd.Series)
        assert(set(np.unique(Y_global))==set([0,1])), "Expected binary labels of 0/1"
        assert(X.shape[0]==len(Y_global)), "The dimensions of X and y do not match"
        
        self.explanations_database = {} # stores every computed explanation        
        self.X_original = X
        self.Y_global = Y_global.copy()        
        if set(np.unique(X.values))==set([0,1]):
            self.data_type = "BINARY"            
        else:
            self.data_type = "CONTINUOUS"

        if complement_features:
            # augment data 
            if self.data_type == "BINARY":
                self.X = complement_binary_dataframe(X)
            else:
                self.X = complement_continuous_dataframe(X) 
        else:
            self.X = X.copy()
        self.features = self.X.columns.values.tolist()
        
        self.n, self.p = X.shape
        if DEBUG and 0:
            print('Initialized %s dataset'%self.data_type)
            #print(pd.concat([self.X_original,self.Y_global],axis=1))
            #print(pd.concat([self.X_expanded,self.Y_global],axis=1))

        self.n_models = 0 # counter for the number of models generated by the object
            
    
    def explain(self, X_e, Y_e, objective='SPARSITY', n_explanations=1, max_features=999999, max_runtime=60):
        """
        Generate summary-explanations for observation (x_e,y_e).

        Parameters
        ----------
        X_e : pd.DataFrame
            Observations to explain.
        Y_e : array/list/series of binary values.
            Predictions by a global model.
        objective : string            
            Takes the value "SPARSITY" or "SUPPORT".
        max_features : integer            
            Maximal number of features (relevant for optimizing support).            
        n_explanations : integer
            Maximal number of summary-explanations to generate.        
        max_runtime : integer
            Maximal running time per explanation (in minutes, total is max_runtime*n_explanations).
            
        Returns
        -------
        Dataframe that holds the resulting rules ["Rule", "Prediction", "Support","n _features", "Time", "Algorithm", "x_e"].
        """        
        
        if self.data_type=="BINARY":
            X_e = complement_binary_dataframe(X_e) 
        else:
            X_e = complement_continuous_dataframe(X_e) 
        
        assert(len(np.setdiff1d(Y_e, [0,1]))==0), 'currently supports binary datasets'
        assert(len(X_e)==len(Y_e)), 'Mismatch between the number of observations and predictions'
        assert(type(X_e)==pd.DataFrame), 'Expected dataframe for X_e'
        assert(set(self.features) == set(X_e.columns.values))
        assert(objective in ['SPARSITY','SUPPORT']), 'Unknown objective passed to explain_local'
        assert(n_explanations>=1), 'n_explanations should be >= 1'
        assert(max_features>=1), 'max_features should be >= 1'
        
        res = []
        for i in range(len(X_e)):
            df_i = self.__explain_local__(X_e.iloc[i], Y_e[i], objective, n_explanations, max_features, max_runtime)
            df_i.insert(0, "#Observation",i)
            df_i.insert(1, "#Explanation",df_i.index.values)
            #df_i.index = map(lambda x:"Obs. #%s, Exp. #%s:"%(str(i),str(x)),df_i.index.values)
            res.append(df_i)
        return(pd.concat(res).reset_index(drop=True))
        
        
    
    def __explain_local__(self, x_e, y_e, objective='SPARSITY', n_explanations=1, max_features=999999, max_runtime=60):
        """
        Generate summary-explanations for observation (x_e,y_e).

        Parameters
        ----------
        x_e : pd.Series
            Observation to explain.
        y_e : integer (0/1)
            Prediction by a global model.
        objective : string            
            Takes the value "SPARSITY" or "SUPPORT".
        max_features : integer            
            Maximal number of features (relevant for optimizing support).            
        n_explanations : integer
            Maximal number of summary-explanations to generate.        
        max_runtime : integer
            Maximal running time per explanation (in seconds, total is max_runtime*n_explanations).
            
        Returns
        -------
        Dataframe that holds the resulting rules ["Rule", "Prediction", "Support","n _features", "Time", "Algorithm", "x_e"].
        """        
        
        assert(y_e in [0,1]), 'currently supports binary datasets'
        assert(type(x_e)==pd.Series)
        assert(set(self.features) == set(x_e.index.values))
        assert(objective in ['SPARSITY','SUPPORT']), 'Unknown objective passed to explain_local'
        assert(n_explanations>=1), 'n_explanations should be >= 1'
        assert(max_features>=1), 'max_features should be >= 1'
       
        if self.data_type=="BINARY":
            if objective=="SPARSITY":
                ce_list = self.__BinMinSetCover__(x_e, y_e, n_explanations) 
                #print(ce_list)
            else:
                ce_list = self.__BinMaxSupport__(x_e, y_e, max_features, n_explanations)
        elif self.data_type=="CONTINUOUS":
            ce_list = self.__ContMinSetCover__(x_e, y_e, n_explanations)
            if objective=="SUPPORT":
                ce_list = self.__ContMaxSupport__( x_e, y_e, ce_list, n_binary_solutions = 1, n_best = n_explanations)
                
        return(ConsistentExplanation_list_to_df(ce_list))        
    
    
    
    
    def __str__(self): 
        '''
        Return string representation of the ConsistentRulesExplainer
        '''       
        d = {'about':'this object can generate globally-consistent rule-based explanations for model predictions',
             'features':self.features,
             '(n,p)': (self.n, self.p),
             'data_type':self.data_type}                
        return(str(d))


    def __BinMinSetCover__(self, x_e, y_e, n_explanations=1):
        '''
        Generate sparse summary-explanations for observatoin (x_e,y_e). Return n_explanations distinct summary-explanations.
        
        x_e : pd.Series
        y_e : model prediction for observation x_e
        n_explanations : the maximal number of solutions to be returned

        output: list of ConsistentExplanation, explanations
        '''
        start = time.time()
        assert(y_e in [0,1]), 'currently supports binary datasets'
        assert(type(x_e)==pd.Series)
        assert(set(self.features) == set(x_e.index.values))
        assert(self.data_type=="BINARY")

        solutions = []
        res = []
        for counter in range(n_explanations):            
            self.n_models+=1
            opt_model = pulp.LpProblem("BinMinSetCover-%d"%self.n_models, pulp.LpMinimize)
            b = pulp.LpVariable.dicts('b', self.features, 
                                      lowBound = 0,
                                      upBound = 1,
                                      cat = pulp.LpInteger)

            for s_index, s in enumerate(solutions):
                opt_model += (sum([b[f] for f in self.features if s[f]==0]) + sum([1-b[f] for f in self.features if s[f]==1]) >=1), "Prohibit attaining previous solution %d"%s_index
                
            for f in self.features:
                if x_e[f]==0:
                    opt_model += (b[f] == 0), "Feature %s must be 1"%f
    
            opt_model += sum([b[f] for f in self.features]) # objective function
            X_other = self.X.loc[self.Y_global!=y_e]
            X_same = self.X.loc[self.Y_global==y_e]
            for index, row in X_other.iterrows():
                opt_model += sum([b[f] for f in self.features if row[f]==0]) >= 1, "Const. obs. %s"%str(index)
                
            #opt_model.writeLP("[%s]model(BinMinSetCover).txt"%get_time())
            #status = opt_model.solve(solver = GUROBI(msg=False, OutputFlag=0,TIME_LIMIT=TIME_LIMIT))
            status = opt_model.solve(solver = GUROBI(msg=False, OutputFlag=0,TIME_LIMIT=TIME_LIMIT))
            if (status==pulp.constants.LpStatusInfeasible):
                break
            
            tau = pd.Series(-np.inf, index=self.features)            
            solutions.append({})
            for f in self.features:                 
                solutions[-1][f]=b[f].value()
                if b[f].value()==1:
                    tau[f] = 1                 
            rbm = ConsistentExplanation(tau,{'time':'%.1f'%(time.time()-start), 'type':'BinMinSetCover'})   
            
            rbm.set_support(self.X, self.Y_global, y_e)
            rbm.set_explained_observation(x_e, y_e)
            if rbm.properties['support']==-1:
                break
            res.append(rbm)

        #res = list(filter(lambda x:x is None, res))   WHY DID I ADD IT?          
        return(res)


   
    def __BinMaxSupport__(self, x_e, y_e, max_features, n_explanations):
        
        '''
        Generates a single explanation with maximal support subject to constraint
        on the maximal number of features used in the explanation.
        
        x_e : pd.Series (compatible with original feature space)
        y_e : binary prediction
        max_features : maximal number of terms
        
        output: ConsistentExplanation, an explanation (just one)
        '''
        start = time.time()
        assert(y_e in [0,1]), 'currently supports binary datasets'
        assert(type(x_e)==pd.Series)
        assert(set(self.features) == set(x_e.index.values))
        assert(self.data_type=="BINARY")


        X_other = self.X.loc[self.Y_global!=y_e]
        X_same = self.X.loc[self.Y_global==y_e]
        same_indexes = X_same.index.values.tolist()

        solutions = []
        res = []
        for counter in range(n_explanations):            
            self.n_models+=1
            opt_model = pulp.LpProblem("BinMaxSupprt-%d"%self.n_models, pulp.LpMaximize)
            
            # decision variables
            b = pulp.LpVariable.dicts('b', self.features, 
                                      lowBound = 0,
                                      upBound = 1,
                                      cat = pulp.LpInteger)
            r = pulp.LpVariable.dicts('r', same_indexes, 
                                      lowBound = 0,
                                      upBound = 1,
                                      cat = pulp.LpInteger)

            for s_index, s in enumerate(solutions):
                opt_model += (sum([b[f] for f in self.features if s[f]==0]) + sum([1-b[f] for f in self.features if s[f]==1]) >=1), "Prohibit attaining previous solution %d"%s_index
                
            # objective function
            opt_model += sum([r[i] for i in same_indexes]) 
    
            for f in self.features:
                if x_e[f]==0:
                    opt_model += (b[f] == 0), "Feature %s must be 0"%f
    
            for index, row in X_other.iterrows():
                opt_model += sum([b[f] for f in self.features if row[f]==0]) >= 1, "Rule doesn't apply to obs. %s"%str(index)
                
            for index, row in X_same.iterrows():
                opt_model += sum([b[f] for f in self.features if row[f]==0]) <= self.p*(1-r[index]), "Const. obs. %s"%str(index)
    
            opt_model += (sum([b[f] for f in self.features]) <= max_features), "Max. num. of features"
            #opt_model.writeLP("model(BinMaxSupport).lp")
            
            status = opt_model.solve(solver = GUROBI(msg=False, OutputFlag=0,TIME_LIMIT=TIME_LIMIT))        
            if (status==pulp.constants.LpStatusInfeasible):
                break
              
                        
            tau = pd.Series(-np.inf, index=self.features)
            solutions.append({})
            for f in self.features: 
                solutions[-1][f]=b[f].value()
                if b[f].value()==1:
                    tau[f] = 1                 
            rbm = ConsistentExplanation(tau,{'time':'%.1f'%(time.time()-start), 'type':'BinMaxSupport'})          
            rbm.set_support(self.X, self.Y_global, y_e)
            rbm.set_explained_observation(x_e, y_e)
            if rbm.properties['support']==-1:
                break
            res.append(rbm)
            assert(rbm.predict(np.array(x_e).reshape((1,-1)))), 'Generated explanation does not agree with explained observation'
            
        return(res)




    
    def generate_explanations_for_training_data(self, obs_list, explanation_types = {}):
        '''
        Generate explanations from the training data. All generated explanations
        are stored internally and are reused if later requested. 
        
        obs_list : list, collection of observation indexes for which 
                   explanations are to be created.
                   
        explanation_types : dictionary, each key is an explanation type and 
                            each value is a dictionary with parameters values. 
                            Examples:
                            {'BinMinSetCover':{'n_solutions':3}}  
                            {'BinMaxSupport':{'n_terms':1}}  # n_terms=1
                            {'BinMaxSupport':{'n_terms':'minimal+1'}} # n_terms= best acheives using BinMaxSetCover+1
                            
        
        '''
        assert(max(obs_list)<self.n)
        assert(min(obs_list)>=0)
        res = []
        for i in obs_list:                                   
            x, y_m = self.X_original.iloc[i], self.Y_global[i]
            for e_type in explanation_types:
                if DEBUG:
                    print("generate_explanations_for_training_data", i, e_type)
                    
                if e_type=='BinMinSetCover':
                    n_solutions = explanation_types['BinMinSetCover']['n_solutions']                    
                    e_list = self.explanations_database.get((i,'BinMinSetCover'), [])
                    if e_list==[]: 
                        e_list = self.__BinMinSetCover__(x, y_m, n_solutions) 
                        self.explanations_database[(i,'BinMinSetCover')] = e_list
                    for e in e_list:
                        e.properties['function_call'] = (i,'BinMinSetCover')
                        e.properties['observation'] = i
                    res.append(e_list)
                    
                elif e_type=='BinMaxSupport':                                
                    n_terms = explanation_types['BinMaxSupport']['n_terms']                    
                    n_terms_str = str(n_terms)
                    e = self.explanations_database.get((i,'BinMaxSupport',n_terms_str), None)
                    if e==None:                    
                        if n_terms_str.startswith('minimal'):
                            e_list = self.explanations_database.get((i,'BinMinSetCover'), [])
                            if e_list==[]: 
                                e_list = self.__BinMinSetCover__(x, y_m, n_solutions=1)
                                self.explanations_database[(i,'BinMinSetCover')] = e_list                                                
                            n_terms = e_list[0].properties['n_terms'] + int(n_terms_str.split("+")[1]) # extract number from "minimal+3"
                        e = self.__BinMaxSupport__(x, y_m, n_terms)
                        if e==None: 
                            res.append(None)
                        else:
                            self.explanations_database[(i,'BinMaxSupport',n_terms_str)] = e
                            e.properties['function_call'] = (i,'BinMaxSupport',n_terms_str)
                            e.properties['observation'] = i
                            res.append(e)
                
                elif e_type=='ContMinSetCover':
                    n_solutions = explanation_types['ContMinSetCover']['n_solutions']   
                    e_list = self.explanations_database.get((i,'ContMinSetCover',n_solutions), [])
                    if e_list==[]: 
                        e_list = self.__ContMinSetCover__(x, y_m, n_solutions) 
                        self.explanations_database[(i,'ContMinSetCover')] = e_list
                        for e in e_list:
                            e.properties['function_call'] = (i,'ContMinSetCover')
                            e.properties['observation'] = i
                    res.append(e_list)
    
                elif e_type=='ContMaxSupport':                                
                    n_solutions = explanation_types['ContMaxSupport']['n_solutions']   
                    e_list = self.explanations_database.get((i,'ContMaxSupport',n_solutions), [])
                    if e_list==[]:
                        e_list_min = self.explanations_database.get((i,'ContMinSetCover',n_solutions), [])
                        if e_list_min==[]:                    
                            e_list_min = self.__ContMinSetCover__(x, y_m, n_solutions) 
                            self.explanations_database[(i,'ContMinSetCover')] = e_list
                            for e in e_list_min:
                                e.properties['function_call'] = (i,'ContMinSetCover')
                        e = self.__ContMaxSupport__( x, y_m, e_list_min)    
                        e.properties['function_call'] = (i,'ContMaxSupport',n_solutions)
                        e.properties['observation'] = i
                    res.append(e)

                else:
                    raise Exception('Explanation type is not supported:', e_type)                                                    
                
                if DEBUG:
                    pass
                    #print(e)
                print(explanation_types[e_type],res[-1].__repr__(),'\n')                
        return(res)
            
                




        
    def __ContMinSetCover__(self, x_e, y_e, n_binary_solutions=1):
        '''
        Returns sprase explanations for continuous datasets. The algorithm 
        converts the continuous dataset in a binary one (see the paper) and
        then applied BinMinSetCover to return n_binary_solutions solutions.
        
        x_e : observation to be explained
        y_e : corresponding label
        n_binary_solutions : the maximal number of solutions to be returned
        '''
    
        assert(y_e in [0,1]), 'currently supports binary datasets'
        assert(type(x_e)==pd.Series)
        assert(set(self.features) == set(x_e.index.values))
        assert(self.data_type=="CONTINUOUS")
        assert_cont_expanded(self.X), "__ContMinSetCover__ only works with continuously expanded datasets (call complement_continuous_dataframe)"
        
        X_other = self.X.loc[self.Y_global!=y_e]
        X_same = self.X.loc[self.Y_global==y_e]
        same_indexes = X_same.index.values.tolist()

        start = time.time()  
        X_bin = (self.X >= x_e).astype(int) # assumes complementing features are in the X
        x_e_bin = (x_e>=x_e).astype(int)        
        bin_GCRBE = ConsistentRulesExplainer(X_bin, self.Y_global, complement_features=False)        
        
        res = []
        rbm_bin_solutions = bin_GCRBE.__BinMinSetCover__(x_e_bin, y_e, n_binary_solutions)
        for rbm_bin in rbm_bin_solutions:        
            tau_cont = pd.Series(-np.inf, index=self.features)
            for f in self.features:
                if np.isfinite(rbm_bin.tau[f]):
                    tau_cont[f]=x_e[f]    
            rbm = ConsistentExplanation(tau_cont,{'time':'%.1f'%(time.time()-start),
                                    'type':'ContMinSetCover'})                      
            rbm.set_support(self.X, self.Y_global, y_e)
            rbm.set_explained_observation(x_e, y_e)
            assert(rbm.predict(np.array(x_e).reshape((1,-1)))), 'Generated explanation does not agree with explained observation'
            res.append(rbm)
        return(res)


    def __ContMaxSupport__(self, x_e, y_e, e_list=[], n_binary_solutions = 1, n_best = 1):
        '''
        Expands given solutions to increase support while maintaining the same features. Returns exactly one explanation         
        
        x_e : observation to be explained
        y_e : corresponding label
        e_list : explanations to expanded (or ContMinSetCover solution used as a starting point for expansion) 
        n_binary_solutions : how many initial solution to create if e_list is empty
        n_best : return the n_best best solutions
        '''
        start = time.time()  
        if e_list==[]: 
            e_list = self.__ContMinSetCover__(x_e, y_e, n_binary_solutions) # call takes care of input validation and find the basis for expansion
        else:   
            assert(y_e in [0,1]), 'currently supports binary datasets'
            assert(type(x_e)==pd.Series)
            assert(set(self.features) == set(x_e.index.values))
            assert(self.data_type=="CONTINUOUS")
            assert_cont_expanded(self.X), "__ContMinSetCover__ only works with continuously expanded datasets (call complement_continuous_dataframe)"
            for e in e_list:
                assert(e.support(self.X, self.Y_global, y_e)>-1), 'provided explanation is not consistent:'+ str(e)           
        res = []
        for e in e_list:
            #tau = pd.Series(index=x_e.index, data=np.concatenate([e.tau, e.ub]))
            tau = e.tau
            thresholds, state, features = {}, [], []
            for f in self.features:            
                if np.isfinite(tau[f]):
                    features.append(f)
                    #thresholds[f] = self.X.loc[(self.Y_global==y_e)&(self.X[f]<=x_e[f]),f].unique().tolist()                
                    thresholds[f] = self.X.loc[:,f].unique().tolist()                
                    thresholds[f].append(-np.inf) 
                    thresholds[f] = sorted(thresholds[f])            
                    state.append(thresholds[f].index(e.tau[f]))                
            #state = np.array(state)        
            self.global_counter=0
            explored_states = {}
            self.solve_DP(tau, features, thresholds, state, y_e, explored_states)                
            e_expanded = max(explored_states.values(), key=lambda e_:e_.properties['support'])
            e_expanded.properties['time'] = '%.1f'%(time.time()-start)
            e_expanded.properties['type'] = 'ContMaxSupport'
            e_expanded.set_explained_observation(x_e, y_e)
            assert(e.predict(np.array(x_e).reshape((1,-1)))), 'Generated explanation does not agree with explained observation'
            res.append(e_expanded)            
        
        res = set(res)
        res = sorted(res, key=lambda e_:e_.properties['support'], reverse=True)
        #e_best = max(res, key=lambda e_:e_.properties['support'])
        return(res[:n_best])


    def solve_DP(self, tau, features, thresholds, state, y_e, explored_states):
        '''
        features : list of strings, features used in the rule
        thresholds : list of numeric values, support of each feature (including +/- np.inf, sorted)
        state : np.array, current state (defined using indexes to 'thresholds' entries)
        y_e : binary, label of the explained observation
        explored_states : set of explored states, prevents repeated visits
        '''
        self.global_counter+=1
            
        if state is None:
            raise Exception('shouldnt be here!')
            
        state_key = tuple(state)
        if state_key in explored_states: 
            #print('returned from here 1')
            return            
        
        tau = tau.copy()
        for f_index, f in enumerate(features):    
            tau[f] = thresholds[f][state[f_index]]
        
        #print(state_key)
        #print('before:\n',explored_states)
        e_curr = ConsistentExplanation(tau,{})    
        e_curr.set_support(self.X,self.Y_global,y_e)        
        #e_curr.set_explained_observation(x, y_e)
        #print('after:\n',e_curr, hex(id(e_curr)))
        explored_states[state_key]=e_curr                    
        if e_curr.properties['support']==-1:
            return            
                        
        if self.global_counter%1000==0:
            print('   ContMaxSupport:solve_DP [',self.global_counter, '] state', state, 'support', e_curr.properties['support'])
            pass
        
        for f_index, f in enumerate(features):               
            if state[f_index]==0:
                continue               
            else:
                state_next = state.copy()
                state_next[f_index]-=1
                self.solve_DP(tau, features, thresholds, state_next, y_e, explored_states)                                    

        
        
        
    
        
    def get_explanations(self):        
        pass
    
    def get_explanations_stats(self):
        pass
        # print how many explanations does the object holds; for all observations? 

    def export_explanations(self, file):
        pass

    def import_explanations(self, file):
        pass
    
    def generate_explanations_new(self, X):
        pass
    





def toy_binary_dataset(plot=True, print_table=True, file_name='toy_data_cont.png'):
    k = 2
    data=np.array([((i,j),i,j,0) for i in range(k) for j in range(k)], dtype=object)
    df = pd.DataFrame(data, columns=['i','x1','x2','y'])    
    df.index = df.iloc[:,0].values
    df = df.iloc[:,1:]
    df.loc[[(0,0)],'y']=1    
    df.reset_index(drop=True, inplace=True)    
    X, y = df[['x1','x2']], df['y']
    if print_table:
        print(df)    
    if plot: 
        fig, ax = plt.subplots()
        df.plot.scatter(x='x1',y='x2',c='y', cmap=cm.get_cmap('Spectral'),sharex=False,s=120, ax=ax)
        plt.savefig(file_name)
    return(X,y,df)

def toy_continuous_dataset(plot=True, print_table=True, file_name='toy_data_cont.png'):
    k = 3
    data=np.array([((i,j),i,j,0) for i in range(10,10+k) for j in range(10,10+k)], dtype=object)
    df = pd.DataFrame(data, columns=['i','x1','x2','y'])    
    df.index = df.iloc[:,0].values
    df = df.iloc[:,1:]
    df.loc[[(10,12)],'y']=1    
    df.loc[[(12,10)],'y']=1    
    df.reset_index(drop=True, inplace=True)    
    X, y = df[['x1','x2']], df['y']
    if print_table:
        print(df)    
    if plot: 
        fig, ax = plt.subplots()
        df.plot.scatter(x='x1',y='x2',c='y', cmap=cm.get_cmap('Spectral'),sharex=False,s=120, ax=ax)
        plt.savefig(file_name)
    return(X,y,df)





if __name__ == "__main__":
    
        
    if 1: 
        # illustrate generating explanations for binary datasets
        X,Y_global,df = toy_binary_dataset(plot=True, print_table=True)
        explainer = ConsistentRulesExplainer(X, Y_global)        
        #print(explainer)        
        X_one, Y_one = X.iloc[[0],:], Y_global.iloc[[0]]        
        df_explanations = explainer.explain(X, Y_global, objective='SUPPORT', n_explanations=10, max_features=9999, max_runtime=60)
        print(df_explanations)
        
    if 1: 
        # illustrate generating continuous for binary datasets
        X,Y_global,df = toy_continuous_dataset(plot=True, print_table=True)
        explainer = ConsistentRulesExplainer(X, Y_global)        
        #print(explainer)        
        X_one, Y_one = X.iloc[[0],:], Y_global.iloc[[0]]        
        #df_explanations = explainer.explain(X, Y_global, objective='SUPPORT', n_explanations=10, max_features=9999, max_runtime=60)
        df_explanations = explainer.explain(X_one, Y_one, objective='SUPPORT', n_explanations=10, max_features=9999, max_runtime=60)
        print(df_explanations)
        
        
        
