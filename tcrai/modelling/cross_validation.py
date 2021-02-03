# Copyright 2021 Regeneron Pharmaceuticals Inc. All rights reserved.
# License for Non-Commercial Use of TCRAI code
# All files in this repository (“source code”) are licensed under the following terms below:
# “You” refers to an academic institution or academically employed full-time personnel only. 
# “Regeneron” refers to Regeneron Pharmaceuticals, Inc.
# Regeneron hereby grants You a right to use, reproduce, modify, or distribute the source code to the TCRAI algorithms, in whole or in part, whether in original or modified form, for academic research purposes only.  The foregoing right is royalty-free, worldwide, revocable, non-exclusive, and non-transferable.  
# Prohibited Uses:  The rights granted herein do not include any right to use by commercial entities or commercial use of any kind, including, without limitation, any integration into other code or software that is used for further commercialization, any reproduction, copy, modification or creation of a derivative work that is then incorporated into a commercial product or service or otherwise used for any commercial purpose, or distribution of the source code not in conformity with the restrictions set forth above, whether in whole or in part and whether in original or modified form, and any such commercial usage is not permitted.  
# Except as expressly provided for herein, nothing in this License grants to You any right, title or interest in and to the intellectual property of Regeneron (either expressly or by implication or estoppel).  Notwithstanding anything else in this License, nothing contained herein shall limit or compromise the rights of Regeneron with respect to its own intellectual property or limit its freedom to practice and to develop its products and product candidates.
# If the source code, whole or in part and in original or modified form, is reproduced, shared or distributed in any manner, it must (1) identify Regeneron Pharmaceuticals, Inc. as the original creator, and (2) include the terms of this License.  
# UNLESS OTHERWISE SEPARATELY AGREED UPON, THE SOURCE CODE IS PROVIDED ON AN AS-IS BASIS, AND REGENERON PHARMACEUTICALS, INC. MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE SOURCE CODE, IN WHOLE OR IN PART AND IN ORIGINAL OR MODIFIED FORM, WHETHER EXPRESS, IMPLIED, STATUTORY, OR OTHER REPRESENTATIONS OR WARRANTIES. THIS INCLUDES, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT KNOWN OR DISCOVERABLE. 
# In no case shall Regeneron be liable for any loss, claim, damage, or expenses, of any kind, which may arise from or in connection with this License or the use of the source code. You shall indemnify and hold Regeneron and its employees harmless from any loss, claim, damage, expenses, or liability, of any kind, from a third-party which may arise from or in connection with this License or Your use of the source code. 
# You agree that this License and its terms are governed by the laws of the State of New York, without regard to choice of law rules or the United Nations Convention on the International Sale of Goods.
# Please reach out to Regeneron Pharmaceuticals Inc./Administrator relating to any non-academic or commercial use of the source code.
""" Module for objects relating to performing cross-validation of BaseXCR models """
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import pandas as pd

from abc import ABC,abstractmethod

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold,StratifiedShuffleSplit,ShuffleSplit
from sklearn.model_selection import train_test_split

DEFAULT_COMPILATION = {
    'optimizer': keras.optimizers.Adam(),
    'loss' : keras.losses.BinaryCrossentropy(),
    'metrics' : [keras.metrics.BinaryAccuracy()]}

DEFAULT_FIT = {
    'epochs' : 200,
    'callbacks' : keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=3, 
                                                restore_best_weights=True),
    'verbose' : 0
}

DEFAULT_COMPILATION_BUILDER = lambda x : DEFAULT_COMPILATION
DEFAULT_FIT_BUILDER = lambda x : DEFAULT_FIT

class CrossValidator:
    """ Perform Cross-validation of a model type
    
    This object can be used to perform e.g K fold cross-val, and monte carlo cross-val, by
    choosing the appropriate `splitter` for the type of cross-validation wanted.
    
    Use the method train() to perform cross-validation of the model type specified by model_builder.
    
    The best performing model is saved in CrossValidator.best_model. One can also access the results
    of cross-validation; the hyperparamters used and the results over each set, in the pandas dataframe
    stored in CrossValidator.results_table. The value of CrossValidator.best_hp_idx specifies the index
    of the row in CrossValidator.results_table which was best performing, and corresponds the 
    hyperparameters used for retraining the model with the full training set (validation data only used
    for early stopping).
    
    parameters
    ------------
    splitter: sklearn.model_selection Splitter class 
            A Splitter class from sklearn.model_selection
            see: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
            e.g. sklearn.model_selection.KFold()        
    n_folds: int
            number of folds
    hp_grid: sklearn.model_selection.ParameterGrid
            A ParameterGrid over hp options
    model_builder: callable
            Single argument callable taking a hyperparameter dict as argument, and
            returning a BaseXCRModelWithProcessors subclass
    compilation_kwargs_builder: callable
            Single argument callable taking a hyperparameter dict as argument, and
            returning a dictionary
    fit_kwargs_builder: callable
            Single argument callable taking a hyperparameter dict as argument, and
            returning a dictionary of fit kwargs expected by models fit method.
    evaluator_dict: dict, optional
            Dictionary of {name: evaluator} pairs. 
    validation_split: float, optional, default=0.15
            fraction of training data sent in to use as a left-out validation set
    seed: int or None
            Integer to seed internal validation split upon. If None, random splitting 
            will differ between runs.
    
    """
    
    def __init__(self, splitter, hp_grid, model_builder,
                  compilation_kwargs_builder=DEFAULT_COMPILATION_BUILDER,
                  fit_kwargs_builder=DEFAULT_FIT_BUILDER,
                  evaluator_dict={'roc':roc_auc_score},
                  validation_split=0.15,
                  seed=None):
        
        self.splitter = splitter
        self.n_folds = self.splitter.get_n_splits()
        self.hp_grid = hp_grid
        self.model_builder = model_builder
        self.evaluator_dict = evaluator_dict
        self.validation_split = validation_split
        
        self.compilation_kwargs_builder = compilation_kwargs_builder
        self.fit_kwargs_builder = fit_kwargs_builder
        
        self.dict_flag = None
        self.label_dict_flag = None
        
        self.best_model = None
        self.results_table = None
        self.best_hp_idx = None
        
        self.seed = seed
        
    def _get_compiled_model(self,hp,inputs,labels):
        model = self.model_builder(hp,inputs,labels)
        for name,e in self.evaluator_dict.items():
            model.add_evaluator(e, name)

        compilation_kwargs = self.compilation_kwargs_builder(hp)
        model.compile(**compilation_kwargs)
        return model
    
    def _retrain_best(self,hp,fit_kwargs,inputs,y_train):
        model = self._get_compiled_model(hp,inputs,y_train)
        self.retrain_history = model.fit( inputs,y_train,**fit_kwargs)
        return model
    
    def _construct_results_table(self,xval_evals):
        """ Using the scores (xval evals) of cross-validation to build results table
        
        parameters
        -----------
        xval_evals: dict
            Dictionary of the results of cross-validation over hyperparams and folds for 
            different evaluation types. The names of the evaluation scores (e.g. 'roc') should
            be the keys, and the values should be np.arrays of shape [N_folds,N_hyperparams]
        """
        cols = self.hp_grid[0].keys()
        hp_data_cols = [[(( 'hyperparameters', c ),hp[c]) for c in cols ] for hp in self.hp_grid]
        
        self.results_table = pd.DataFrame(data=[[r[1] for r in row] for row in hp_data_cols], 
                                          columns = pd.MultiIndex.from_tuples([
                                              ('hyperparameters', c) for c in cols
                                          ]) )
        # now add the results in 
        for name,vals in xval_evals.items():
            mu = np.mean(vals,axis=0)
            sigma = np.std(vals,axis=0)
            results_table_x = pd.DataFrame(data=[[m,s] for m,s in zip(mu,sigma)], 
                              columns = pd.MultiIndex.from_tuples([
                                  (name, 'mu'), (name, 'sigma') 
                              ]) )
            self.results_table = self.results_table.join(results_table_x) 
        return None  
    
    def _train_split(self,idxs,stratify):
        """ get train/(validation or test) split via index
        
        parameters
        ----------
        idxs: np.array or list
            list/array of indices to be split
            
        stratify: np.array or list
            list/array of values on which to stratify the split. If None,
            no stratification will be performed.
            
        returns
        -------
            (train_idxs, test_idxs) : tuple of np.arrays
        """
        if stratify is not None:
            shuffle = StratifiedShuffleSplit(n_splits=1,
                                             test_size=self.validation_split,
                                             random_state=self.seed)
            train_idxs, vali_idxs = next(iter(shuffle.split(idxs, stratify)))
        else:
            shuffle = ShuffleSplit(n_splits=1,
                                   test_size=self.validation_split,
                                   random_state=self.seed)
            train_idxs, vali_idxs = next(iter(shuffle.split(idxs, stratify)))
        return train_idxs,vali_idxs  
    
    def _apply_idxs_on_dict(self,inputs,idxs):
        """select data at indices in idxs for dict like input"""
        new_inputs = dict()
        for k,v in inputs.items():
            new_inputs[k] = v[idxs]
        return new_inputs
    
    def _apply_idxs_on_list(self,inputs,idxs):
        """select data at indices in idxs for list like input"""
        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(inputs[i][idxs])
        return new_inputs
    
    def check_input(self,inputs):
        """check input is of allowed type, and set appropriate flags"""
        if isinstance(inputs,dict):
            self.dict_flag = True
        elif isinstance(inputs,(tuple, list)):
            self.dict_flag = False
        else:
            raise TypeError("inputs should be dict, or list, or tuple")
        return None
    
    def check_label_type(self,labels):
        """ check if labels are dict-like or not"""
        if isinstance(labels,dict):
            self.label_dict_flag = True
        return None
    
    def train_test_split_via_idxs(self,inputs,train_idxs,test_idxs):
        """ split inputs based on train/test indices
        
        this method also performs appropriate splitting for the input type
        
        parameters
        ----------
        inputs: dict or np array -like input 
            the input data
        
        train_idxs: list/np.array
            indices to be used from the data for training
        
        test_idxs: list/np.array
            indices to be used from the data for testing
            
        returns
        -------
        x_train,y_train: tuple of np.arrays or dicts of np.arrays
            dict format iff inputs/labels were dict format
        """
        if self.dict_flag is None:
            self.check_input(inputs)
            
        if self.dict_flag:
            x_train = self._apply_idxs_on_dict(inputs,train_idxs)
            x_test = self._apply_idxs_on_dict(inputs,test_idxs)
        else:
            x_train = self._apply_idxs_on_list(inputs,train_idxs)
            x_test = self._apply_idxs_on_list(inputs,test_idxs)
        return x_train, x_test
    
    def train_vali_idxs(self,inputs,stratify):
        """ get indices for training/validation
        
        Uses _train_split, but checks the input type.
        
        parameters
        -----------
        inputs: dict or list-like
            input data
        
        stratify: np.array or list
            list/array of values on which to stratify the split. If None,
            no stratification will be performed.
            
        returns
        -------
            (idxs_train,idxs_vali) : tuple of np.arrays
            
        """
        if self.dict_flag is None:
            self.check_input(inputs)
        
        if self.dict_flag:
            idxs = np.arange(len(next(iter((inputs.values())))))
            idxs_train,idxs_vali = self._train_split(idxs,stratify)
        else:
            idxs = np.arange(len(inputs[0]))
            idxs_train,idxs_vali = self._train_split(idxs,stratify)
        return idxs_train,idxs_vali
    
    def apply_idxs_to_labels(self,labels,idxs):
        """ get labels only at given indices
        
        Takes into account the input data type
        
        parameters
        ----------
        labels: dict or list like
            The data labels
            
        idxs: list-like
            The indices the labels are wated at
            
        returns
        --------
        out_labels: list-like or dict-like
            The selected labels
        
        """
        if self.label_dict_flag is None:
            self.check_label_type(labels)   
        
        if self.label_dict_flag:
            out_labels = self._apply_idxs_on_dict(labels,idxs)
        else:
            out_labels = labels[idxs]
        return out_labels
    
    def _single_input_example(self,inputs):
        """ Gte the first input data as an example"""
        if self.dict_flag:
            split_ex = next(iter(inputs.values()))
        else:
            split_ex = inputs[0]
        return split_ex
    
    def train(self,inputs,labels,
              sample_weights=None,
              stratify=False,
              stratify_dict_key=None,
              key_evaluation='roc',
              apply_clearing=True):
        """ Train the model using the cross-validation technique of the class
        
        parameters
        -----------
        
        inputs: dict
            dictionary of inputs
        
        labels: model dependent - np.array, list of arrays, dict,..
            labels for each sample
        
        sample_weights: np.array, optional, default=None
            Weights to apply to loss of each specific sample. If None, no weighting will be applied.
        
        stratify: bool, optional, default=False
            Whether to stratify the data by class type
        
        stratify_dict_key: string, optional, default=None
            If the labels are a dictionary, the value of stratify_dict_key will be the 
            key in the label dictionary over which data is stratified.
            
        key_evaluation: string, optional, default = 'roc'
            The key of the internal evaluator dictionary to make the decision to prefer one solution
            over another. default is the area under the ROC curve.
            
        apply_clearing: bool, optional, default=True
            If True, clears the keras session after each model is trained. A final model is trained at the
            end of the cross-validation and kept without clearing the session, so the optimal model is not lost.
            Recommend to keep True, else there is data leak.
            
        returns
        --------
        
        idxs: tuple
            tuple of (idxs_train, idxs_vali) - the indices of the input data that were chosen ad the training
            and validation sets respectively.
            
        """
        
        self.check_input(inputs)
        self.check_label_type(labels)
        
        if stratify and self.label_dict_flag:
            if stratify_dict_key is None:
                raise ValueError("if labels are a dict, you must provide" + 
                                  "stratify_dict_key as the key to be stratified on")
            else:
                stratify_labels = labels[stratify_dict_key]
        elif stratify:
            stratify_labels = labels
        else:
            stratify_labels=None
                
        
        idxs_train,idxs_vali = self.train_vali_idxs(inputs,stratify_labels)
        train_inputs,vali_inputs = self.train_test_split_via_idxs(inputs,idxs_train,idxs_vali)
        
        train_labels = self.apply_idxs_to_labels(labels,idxs_train)
        vali_labels = self.apply_idxs_to_labels(labels,idxs_vali)
        
        if sample_weights is not None:
            train_sample_weights = sample_weights[idxs_train]
            vali_sample_weights = sample_weights[idxs_vali]
        
        if stratify_labels is not None:
            stratify_labels = stratify_labels[idxs_train]
        
        xval_evals = dict()
        for name in self.evaluator_dict.keys():
            xval_evals[name] = np.zeros((self.n_folds, len(self.hp_grid)))
        
        xval_scores = np.zeros((self.n_folds, len(self.hp_grid)))
        
        split_ex = self._single_input_example(train_inputs)
        
        def get_fit_kwargs(hp):
            fit_kwargs = self.fit_kwargs_builder(hp)
            if sample_weights is not None:
                fit_kwargs.update({'validation_data' : ( vali_inputs, vali_labels, vali_sample_weights ) })
                fit_kwargs.update({'sample_weight': sw_train})
            else:
                fit_kwargs.update({'validation_data' : ( vali_inputs, vali_labels ) })
            return fit_kwargs
        
        for i,(train_idxs,test_idxs) in enumerate( self.splitter.split(np.arange(len(split_ex)), stratify_labels) ):
            print("fold ", i+1, " of ", self.n_folds)
            
            x_train,x_test = self.train_test_split_via_idxs(train_inputs,train_idxs,test_idxs) 
            
            y_train = self.apply_idxs_to_labels(train_labels,train_idxs)
            y_test = self.apply_idxs_to_labels(train_labels,test_idxs)
            
            if sample_weights is not None:
                sw_train = train_sample_weights[train_idxs]
                sw_test  = train_sample_weights[test_idxs]

            for j,hp in enumerate(self.hp_grid):
                full_model = self._get_compiled_model(hp,x_train,y_train)
                
                fit_kwargs = get_fit_kwargs(hp)
                
                history = full_model.fit( x_train, y_train, **fit_kwargs)
                
                eval_kwargs = {'sample_weight': None}
                if sample_weights is not None:
                    eval_kwargs.update({'sample_weight': sw_test})
                    
                scores = full_model.evaluate( x_test, y_test, **eval_kwargs )
                for name,val in scores.items():
                    xval_evals[name][i,j] = scores[name]
                    if name==key_evaluation:
                        xval_scores[i,j] = scores[name]
                if apply_clearing:
                    keras.backend.clear_session()
                
        
        mu = np.mean(xval_scores,axis=0)
        sigma = np.std(xval_scores,axis=0)
        best_hp_idx = np.argmax(mu)
        self.best_hp_idx = best_hp_idx
        
        # build a dataframe of the results of the cross validation
        self._construct_results_table(xval_evals)
        
        fit_kwargs = get_fit_kwargs(self.hp_grid[self.best_hp_idx])
        self.best_model = self._retrain_best(self.hp_grid[self.best_hp_idx],
                                             fit_kwargs,
                                             train_inputs,
                                             train_labels )
        
        vali_final_evals = self.best_model.evaluate(vali_inputs,vali_labels,**eval_kwargs)
        print("final validation set evaluations = ", vali_final_evals)
        print("expected test set range : ", 
              mu[best_hp_idx]-sigma[best_hp_idx], 
              mu[best_hp_idx]+sigma[best_hp_idx])
        
        rerun_count=0
        while vali_final_evals[key_evaluation]<mu[best_hp_idx]-2*sigma[best_hp_idx] and rerun_count<2:
            print("unstable final run - retrain try ", rerun_count+1)
            self.best_model = self._retrain_best(self.hp_grid[self.best_hp_idx],
                                             fit_kwargs,
                                             train_inputs,
                                             train_labels )
            vali_final_evals = self.best_model.evaluate(vali_inputs,vali_labels,**eval_kwargs)
            print("new validation set evaluations = ",vali_final_evals)
            rerun_count+=1
        
        
        return idxs_train, idxs_vali
        
        