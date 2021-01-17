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

#!/usr/bin/env python
# coding: utf-8

# Make predictions on TCR-pMHC complexes using TCR info


import os
import sys
from os.path import expanduser

HOME = expanduser("~")

import argparse

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd
import pickle
import json
import random

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

from tcrai.modelling.processing import AASeqProcessor
from tcrai.modelling import extractors, closers
from tcrai.modelling.classification import SeqClassificationModelWithProcessor, VJXCRSeqModel
from tcrai.modelling.cross_validation import CrossValidator
from tcrai.plotting import ml_plots

pd.options.mode.chained_assignment = None

os.environ['PYTHONHASHSEED']=str(42)
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

sns.set(context='talk',palette='bright')

MODEL_TYPE_COLS = {
    'full' : ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRA_cdr3'],
    'alpha' : ['TRA_v_gene','TRA_j_gene','TRA_cdr3'],
    'beta' : ['TRB_v_gene','TRB_j_gene','TRB_cdr3'],
    'seq' : ['TRB_cdr3','TRA_cdr3'],
    'gene' : ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene'],
    'beta_cdr3' : ['TRB_cdr3'],
}

id_list_shared_by_all = ['ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML']
id_list_shared_by_final = ['ELAGIGILTV', 'GILGFVFTL', 'GLCTLVAML','NLVPMVATV'] 
id_list_public_key = [
    'GILGFVFTL',
    'LLWNGPMAV',
    'NLVPMVATV',
    'GLCTLVAML',
    'CINGVCWTV',
    'KLVALGINAV',
    'PKYVKQNTLKLAT',
    'ELAGIGILTV'
]

HP_GRID = {
    'init_bias': [0.0], #ignored unless binary model
    'v_gene_embed_dim': [16],
    'j_gene_embed_dim' : [8],
    'vj_dropout': [0.3],
    'seq_embed_dim': [16],
    'seq_filters': [ [64,128,256] ],
    'seq_kernel_widths': [[5,4,4]],
    'seq_dilations':[[1,1,1]],
    'seq_strides': [[1,3,3]],
    'seq_L2_conv':[0.01],
    'seq_dropout_conv':[0.3],
    'pred_units': [[]],
    'pred_dropout': [0.3],
    'lr':[1.0e-3,5.0e-4,1.0e-4],
    'batch_size': [32,256]
}

HP_GRID_BIG = {
    'init_bias': [0.0], #ignored unless binary model
    'v_gene_embed_dim': [16,32],
    'j_gene_embed_dim' : [8,16],
    'vj_dropout': [0.3],
    'seq_embed_dim': [16],
    'seq_filters': [ [64,128,256] ],
    'seq_kernel_widths': [ [4,4,4], [5,4,4] ],
    'seq_dilations':[[1,1,1]],
    'seq_strides': [[1,3,3],[1,1,1]],
    'seq_L2_conv':[0.01],
    'seq_dropout_conv':[0.3],
    'pred_units': [[],[1024]],
    'pred_dropout': [0.3],
    'lr':[1.0e-3,5.0e-4,1.0e-4],
    'batch_size': [32,256]
}

def make_dir_name(mode,model_type,fname,multinomial_ids,pmhc_do=None):
    """ create an output filename based on the parameters used 
    
    parameters
    -----------
    mode: str
        `multinomial` or `binomial`
        
    model_type: str
        the name of the type of model being used
    
    fname: str
        The underlying name of the dataset being used
    
    multinomial_ids: str
        The name for the types of multinomial pMHCs being included
        
    pmhc_do: str, or None
        str of the pMHC for binomial model, None can be passed if multinomial mode.
        
    returns
    --------
    output_dir_name: str
        A name for the output directory
    
    """
    if mode=='multinomial':
        output_dir_name = 'multinomial'
        output_dir_name += '_' + fname
        multi_str = multinomial_ids.replace("_","-") 
        output_dir_name += '_' + multi_str
        output_dir_name += '_' + model_type
        return output_dir_name
    else:
        output_dir_name = 'binomial'
        output_dir_name += '_' + fname
        output_dir_name += '_' + pmhc_do
        output_dir_name += '_' + model_type
        return output_dir_name
    
    
def make_figure_dir(save_path):
    """ make directory for the figures of the run """
    if save_path is not None:
        if not os.path.exists(os.path.join(save_path,'figs')):
            os.makedirs(os.path.join(save_path,'figs'))
        return os.path.join(save_path,'figs')
    else:
        return None

def plot_roc_binomial(y_test,test_preds, pmhc_do, output_path=None,name=None):
    """ plot an ROC curve for a binomial run """
    fig_dir = make_figure_dir(output_path)
    if fig_dir is not None:
        save_path = os.path.join(fig_dir, name)
    else:
        save_path = None
    ml_plots.plot_roc_binomial(y_test,
                               test_preds,
                               label = pmhc_do,
                               save_path=save_path)
    return None
        
def plot_roc_multinomial(y_test,test_preds,int_pmhc_map,output_path=None,name=None):
    """ plot an ROC curve for a multinomial run """
    fig_dir = make_figure_dir(output_path)
    if fig_dir is not None:
        save_path = os.path.join(fig_dir, name)
    else:
        save_path = None
    ml_plots.plot_roc_multinomial(y_test,
                                  test_preds,
                                  labels = int_pmhc_map,
                                  save_path=save_path)
    return None
    
        
def plot_loss(history,output_path=None,name=None):
    """ plot the training loss """
    fig_dir = make_figure_dir(output_path)
    if fig_dir is not None:
        save_path = os.path.join(fig_dir, name)
    else:
        save_path = None
    ml_plots.plot_loss(history,save_path=save_path)
        
def get_model_builder(mode):
    """ Get a function that builds an (untrained) predictive model 
    
    parameters
    -----------
    mode: str
        `multinomial` or `binomial`
        
    returns
    --------
    build_model: function
        A function that takes (hp, inputs, labels) as arguments,
        and returns a tcrai.modelling.SeqClassificationModelWithProcessor
        object.
    
    """
    def build_model(hp,inputs,labels):
        """ Build a sequence predictive model
        
        parameters
        -----------
        
        hp: dict
            A dictionary of input hyperparameters: keys:
                'init_bias': Initial bias for the final layer, if binomial model
                'max_len_h': Maximum length of TRB_CDR3 sequences
                'max_len_l': Maximum length of TRA_cdr3_sequences
                'v_gene_embed_dim':  Embedding dimension for v genes
                'j_gene_embed_dim' : Embedding dimension for j genes
                'vj_dropout': dropout to be used in the gene feature extractor
                'seq_embed_dim': embedding dimension for amino acids
                'seq_filters': list: list of length (number of convolutional layers), 
                                     with each entry being the number of filters for the i`th layer 
                'seq_kernel_widths': list: list of length (number of convolutional layers), 
                                     with each entry being the width of the kernel for the i`th layer 
                'seq_dilations': list: list of length (number of convolutional layers), 
                                 with each entry being dilation used for the i`th layer 
                'seq_strides': list: list of length (number of convolutional layers), 
                               with each entry being the stride used by the kernel for the i`th layer 
                'seq_L2_conv': L2 regularization penalty for the convolutional layers
                'seq_dropout_conv': Dropout fraction following each convolutional layer
                'pred_units':  list: list of length (number of fully-connected layers to use on TCR fingerprint)
                                    each entry is the number of units required in the i`th fully connected layer.
                'pred_dropout': Dropout to use after each filly-connected layer
                'input_list': The names of the models inputs that we want the model to include when constructing the 
                              fingerprint, prior to the final fully connected layers.
                'n_classes': number of output classes - only used if mode==multinomial.
        
        inputs: Not required
                Here for consistency across classes.
        
        labels: Not required
                Here for consistency across classes.
        
        """

        processor_h = AASeqProcessor(hp['max_len_h'])
        processor_l = AASeqProcessor(hp['max_len_l'])

        vj_cols = ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene'] 
        vj_tokenizers = dict()
        for col in vj_cols:
            vj_tokenizers[col] = Tokenizer(filters='',lower=False,oov_token='UNK')
            vj_tokenizers[col].fit_on_texts(inputs[col])

        vj_encs = dict()
        embed_dim = {
            'TRB_v_gene': hp['v_gene_embed_dim'],
            'TRB_j_gene': hp['j_gene_embed_dim'],
            'TRA_v_gene': hp['v_gene_embed_dim'],
            'TRA_j_gene': hp['j_gene_embed_dim']
        }

        for v in embed_dim.keys():
            hp_vj={
                'vj_width':len(vj_tokenizers[v].word_index) +1,
                'vj_embed': embed_dim[v],
                'dropout': hp['vj_dropout']
            }
            vj_encs[v] = extractors.vj_extractor(hp_vj,name=v+'_enc')

        hp_seq = {
            'embed_dim': hp['seq_embed_dim'],
            'filters': hp['seq_filters'],
            'kernel_widths': hp['seq_kernel_widths'],
            'dilations': hp['seq_dilations'],
            'strides': hp['seq_strides'],
            'L2_conv': hp['seq_L2_conv'],
            'dropout_conv': hp['seq_dropout_conv'],
        }
        
        l_seq_enc = extractors.conv_seq_extractor(hp_seq,hp['max_len_l'],processor_l.vocab_size)
        h_seq_enc = extractors.conv_seq_extractor(hp_seq,hp['max_len_h'],processor_h.vocab_size)

        hp_pred = {
            'units': hp['pred_units'],
            'dropout': hp['pred_dropout'],
            'L2_dense': 0.0,
            'init_bias': hp['init_bias'],
            'n_classes': hp['pred_n_classes']
        }

        if mode=='multinomial':
            predictor = closers.make_categorizer(hp_pred)
        else:

            predictor = closers.make_predictor(hp_pred)

        processors = {
            'TRB_cdr3':processor_h,
            'TRA_cdr3':processor_l,
        }

        seq_encoders = {
            'TRB_cdr3':h_seq_enc,
            'TRA_cdr3':l_seq_enc
        }


        model0 = VJXCRSeqModel(seq_encoders,
                               vj_encs,
                               predictor,
                               input_list=hp['input_list']
                              )

        model = SeqClassificationModelWithProcessor(model0, processors=processors, extra_tokenizers=vj_tokenizers)
        return model
    return build_model

def get_compilation_kwargs_builder(mode):
    """Get a function that returns compilation options 
    
    parameters
    ----------
    mode: str
        `binomial` or `multinomial`
    
    returns
    --------
    compilation_kwargs_builder: function
        Function that returns a dictionary of arguments for model 
        comilation.
    
    """
    def compilation_kwargs_builder(hp):
        """ Get args for model compilation
        
        parameters
        -----------
        hp: dictionary of hyperparemeters
        
        returns
        --------
        kwargs: dict
            dict of the arguments for compilation
        """
        kwargs = dict()
        kwargs['optimizer'] = keras.optimizers.Adam(hp['lr'])
        if mode=='multinomial':
            kwargs['loss'] = keras.losses.SparseCategoricalCrossentropy()
        else:
            kwargs['loss'] = keras.losses.BinaryCrossentropy()
        kwargs['metrics'] = [] #[keras.metrics.AUC(name='auc')]
        return kwargs
    return compilation_kwargs_builder

def get_fit_kwargs_builder(weights,verbose=0):
    """Get a function that returns model fitting options 
    
    parameters
    ----------
    weights: dict
        dictionary of class weights
        
    verbose: int, optional, default=0
        Verbosity of training output - see keras.Model documentation
    
    returns
    --------
    fit_kwargs_builder: function
        Function that returns a dictionary of arguments for model 
        fitting.
    
    """
    def fit_kwargs_builder(hp):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True )
        fit_kwargs = {
            'epochs' : 250,
            'callbacks' : [early_stop],
            'class_weight' : {i:weights[i] for i in range(len(weights))},
            'batch_size': hp['batch_size'],
            'verbose': verbose
        }
        return fit_kwargs
    return fit_kwargs_builder


def df_to_input(df):
    """ convert a dataframe into a dictionary of inputs """
    cols = ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRA_cdr3']
    x = { c:df[c].values for c in cols}
    return x

def assign_labels(df,mode,multinomial_ids, pmhc_do):
    """ For the data, assign labels for binding """
    if mode=='multinomial':
        if multinomial_ids=='shared_by_final':
            pmhc_int_map = {pmhc:i for i,pmhc in enumerate(id_list_shared_by_final) }
            int_pmhc_map = {i:pmhc for i,pmhc in enumerate(id_list_shared_by_final) }
        elif multinomial_ids=='shared_by_all':
            pmhc_int_map = {pmhc:i for i,pmhc in enumerate(id_list_shared_by_all) }
            int_pmhc_map = {i:pmhc for i,pmhc in enumerate(id_list_shared_by_all) }
        elif multinomial_ids=='public_key':
            pmhc_int_map = {pmhc:i for i,pmhc in enumerate(id_list_public_key) }
            int_pmhc_map = {i:pmhc for i,pmhc in enumerate(id_list_public_key) }
        else:
            pmhc_int_map = {pmhc:i for i,pmhc in enumerate(df['id'].unique()) }
            int_pmhc_map = {i:pmhc for i,pmhc in enumerate(df['id'].unique()) }

        df['labels'] = df['id'].map(pmhc_int_map)
        return pmhc_int_map,int_pmhc_map 
    else:
        df['labels'] = df['binds']
        return None,None

def get_runtime_df_labels(data_root,
                          mode,
                          model_type,
                          pmhc_do,
                          multinomial_ids,
                          max_len_h,
                          max_len_l):
    """ select the data to be used for this experiment and apply data labels """
    df = pd.read_csv(data_root)

    shared_ids=None
    if mode=='multinomial':
        df = df[df['binds']==1]
        if multinomial_ids=='shared_by_final':
            df = df[df['id'].isin(id_list_shared_by_final)]
        elif multinomial_ids=='shared_by_all':
            df = df[df['id'].isin(id_list_shared_by_all)]
        elif multinomial_ids=='public_key':
            df = df[df['id'].isin(id_list_public_key)]
    else:
        df = df[df['id']==pmhc_do]
    
    df = df[df['TRA_cdr3'].map(lambda x: len(x))<max_len_h]
    df = df[df['TRB_cdr3'].map(lambda x: len(x))<max_len_l]

    _,_ = assign_labels(df,mode,multinomial_ids, pmhc_do)
    df = df.drop_duplicates(subset=MODEL_TYPE_COLS[model_type]+['labels'])
    pmhc_int_map, int_pmhc_map = assign_labels(df,mode,multinomial_ids, pmhc_do)
    labels = df['labels']
    return df,labels,pmhc_int_map, int_pmhc_map
    
def run_single_case(data_path,
                    output_path,
                    mode = 'multinomial',
                    pmhc_do = None,
                    model_type = 'full',
                    folds = 10,
                    multinomial_ids = None,
                    seed = 42):
    """ Run a single case of the model
    
    For a given dataset, and choice to run a multinomial or binomial model, and accordingly 
    a selection of the pmhc to train over, or the set of pmhcs to train over (multinomial), train a 
    predictive model, and save the model, and the results of that model. MCCV cross validation will be 
    performed with `folds` number of folds. 
    
    parameters
    -----------
    data_path: str
        Path to the data to be modelled
        
    output_path: str
        Path for where to save data
        
    mode: str
        `multinomial` or `binomial`
        
    pmhc_do: str
        The pmhc to build model for, if in binomial mode
        
    model_type: str, optional , default=`full`
        options are
         - full : V/J genes + CDR3 of alpha / beta chains
         - alpha : V/J genes + CDR3 of alpha chain only 
         - beta : V/J genes + +CDR3 of beta chain only 
         - seq : CDR3 of alpha and beta chains 
         - gene : V/J genes of alpha and beta chains only 
         - beta_cdr3: CDR3 of the beta chain only 
    
    folds: int, optional, default=10
        Number of folds to use in the MCCV
        
    multinomial_ids: str, optional, default=None
        Must be provided in multinomial mode. The name for the types of multinomial pMHCs being included
    
    seed: int or None, optional, default=42
        If None, the model's initial state, and dataset splitting will be performed randomly without a seed,
        making results not reproducible.
    
    """
    if seed is not None:
        tf.random.set_seed(seed)
        
    fname_basis = data_path.split(os.sep)[-1].split('.')[0]
    data_root = os.path.join(data_path)
    
    # maximum length TCRs we will consider
    max_len_h = 40
    max_len_l = 40
    
    hp_grid = HP_GRID
    hp_grid['max_len_h'] = [max_len_h]
    hp_grid['max_len_l'] = [max_len_l]
    
    if output_path is not None:
        output_path = os.path.join(output_path,make_dir_name(mode,
                                                             model_type,
                                                             fname_basis,
                                                             multinomial_ids,
                                                             pmhc_do=pmhc_do))
        print('output path is :', output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        print("---- will not save ----- ")
    
    df,labels,pmhc_int_map,int_pmhc_map = get_runtime_df_labels(data_root,
                                                               mode,
                                                               model_type,
                                                               pmhc_do,
                                                               multinomial_ids,
                                                               max_len_h,
                                                               max_len_l)
    
    n_classes = len(df['labels'].unique())
    print("number of classes = ", n_classes)

    # add the number of classes to the hyperparameters of the model
    hp_grid['pred_n_classes'] = [n_classes]

    input_list = MODEL_TYPE_COLS[model_type]
    hp_grid['input_list'] = [input_list]
    
    # construct a grid over all hyperparameters
    param_grid = ParameterGrid(hp_grid)

    print("\n number of different hyperparamter combos = ",len(param_grid), '\n')
    
    # split the train and test sets
    train_df,test_df,y_train,y_test = train_test_split(df,
                                                       labels,
                                                       test_size=0.1,
                                                       stratify=labels,
                                                       random_state=seed)

    # Because the model will incorporate the sequence processing, and gene tokenizers,
    # model can directly recieve the strings for these values, not the encodings
    train_in = df_to_input(train_df)
    test_in = df_to_input(test_df)

    y_train = np.expand_dims(y_train.values,axis=1)
    y_test = np.expand_dims(y_test.values, axis=1)
    
    print("\n training data contains ", len(train_df), "samples", '\n')
   
    # Function for building Model
    # this function gets passed later to the cross validator object, 
    # allowing models to be built from different sets of hyperparameters. 
    build_model = get_model_builder(mode)
    # function for hyperparametrized compilation arguments
    compilation_kwargs_builder = get_compilation_kwargs_builder(mode)
    
    weights = compute_class_weight('balanced',classes=np.arange(len(np.unique(y_train))),y=np.squeeze(y_train))
    fit_kwargs_builder = get_fit_kwargs_builder(weights)
    print(weights)
    
    def get_key_evaluator(mode):
        if mode=='multinomial':
            return {'roc' : lambda y_true, y_pred : metrics.roc_auc_score(y_true, 
                                                                             y_pred, 
                                                                             multi_class='ovr', 
                                                                             average='weighted')
                    }

        else:
            return {'roc': metrics.roc_auc_score}

    trainer = CrossValidator(StratifiedShuffleSplit(n_splits=folds,
                                                    test_size=0.15,
                                                    random_state=seed),
                             param_grid,
                             build_model,
                             compilation_kwargs_builder=compilation_kwargs_builder,
                             fit_kwargs_builder=fit_kwargs_builder,
                             evaluator_dict = get_key_evaluator(mode),
                             seed=seed
                            )

    try:
        train_idxs, vali_idxs = trainer.train(train_in,y_train,
                                              stratify=True,
                                              key_evaluation=list(get_key_evaluator(mode).keys())[0]
                                             )
    except ValueError:
        print(" \n \n Ignoring this case  - presumably too few postive examples in the cross-val test set \n \n")
        return None

    # view the table of results 
    # see the hyperparameters used and the mean/std of the ROC (multiclass, weighted, one-vs-rest) across the folds
    print( trainer.results_table )
   
    # add further metric - in addition to ROC 
    if mode=='binomial':
        trainer.best_model.add_evaluator(metrics.average_precision_score,'AP')
    else:
        trainer.best_model.add_evaluator(lambda y_true,y_pred: 
                                         metrics.average_precision_score(LabelBinarizer().fit_transform(y_true),
                                                                         y_pred,
                                                                         average='weighted'),
                                         'AP')
    # evaluate model on the test data
    test_evals = trainer.best_model.evaluate( test_in, y_test )
    print( test_evals )

    # ## Run the model over the test set
    # Then go ahead and calculate/plot the ROC
    test_preds = trainer.best_model.run(test_in)


    if mode=='multinomial':
        plot_roc_multinomial(y_test,
                             test_preds,
                             int_pmhc_map,
                             output_path=output_path,
                             name = 'roc_test.png')
    else:
        plot_roc_binomial(y_test,
                          test_preds,
                          pmhc_do,
                          output_path=output_path,
                          name = 'roc_test.png')

    plot_loss(trainer.retrain_history,output_path=output_path,name='loss_plot.png')

    
    true_train_df = train_df.iloc[train_idxs]
    validation_df = train_df.iloc[vali_idxs]
    
    if mode=='binomial':
        true_train_df.loc[:,'preds'] = trainer.best_model.run(df_to_input(true_train_df))
        validation_df.loc[:,'preds'] = trainer.best_model.run(df_to_input(validation_df))
        test_df.loc[:,'preds'] = test_preds
    else:
        def preds_onto_df(df_):
            preds_ = trainer.best_model.run(df_to_input(df_))
            pred_list = preds_.tolist()
            pred_strs = ['_'.join(["{:.9f}".format(y) for y in pp]) for pp in pred_list]
            df_.loc[:,'preds'] = pred_strs
            return df_
        true_train_df = preds_onto_df(true_train_df)
        validation_df = preds_onto_df(validation_df)
        test_results_df = preds_onto_df(test_df) 

    if output_path is not None:
        trainer.best_model.save(os.path.join(output_path,'model'))
        trainer.results_table.to_csv(os.path.join(output_path,'cross_val_table.csv'))

        (true_train_df[['TCR_id','labels','preds']]
         .set_index('TCR_id')
         .to_csv(os.path.join(output_path,'train_results.csv'))
        )
        (validation_df[['TCR_id','labels','preds']] 
        .set_index('TCR_id')
        .to_csv(os.path.join(output_path,'validation_results.csv'))
        )
        (test_df[['TCR_id','labels','preds']]
         .set_index('TCR_id')
         .to_csv(os.path.join(output_path,'test_results.csv'))
        )
        
        with open(os.path.join(output_path,'test_evals.json'), 'w', encoding='utf8') as f:
            json.dump(test_evals,f,indent=4)
            
        if mode=='multinomial':
            with open(os.path.join(output_path,'int_pmhc_map.pickle'), 'wb') as f:
                pickle.dump(int_pmhc_map, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output_path,'pmhc_int_map.pickle'), 'wb') as f:
                pickle.dump(pmhc_int_map, f, pickle.HIGHEST_PROTOCOL)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='path to data',
                        )
    parser.add_argument('-o',
                        type=str,
                        help='path at which to save data, if not provided, '+
                             'outputs to terminal and plots shown, but nothing saved.',
                        dest='dir_out'
                        )
    parser.add_argument('-mode',
                        type = str,
                        help = ' `binomial` or `multinomial`, default=multinomial ',
                        dest = 'mode')
    parser.add_argument('-pmhc',
                        type = str,
                        help = 'specific pmhc code to train model for if in binomial mode' 
                                +' - fails if not provided when running in binomial mode',
                        dest = 'pmhc_do'
                       )
    parser.add_argument('-model_type',
                        type = str,
                        help = 'specific inputs only options are \n'+
                               ' full : V/J genes + CDR3 of alpha / beta chains \n'+
                               ' alpha : V/J genes + CDR3 of alpha chain only \n'+
                               ' beta : V/J genes + +CDR3 of beta chain only \n'+
                               ' seq : CDR3 of alpha and beta chains \n'+
                               ' gene : V/J genes of alpha and beta chains only \n'+
                               ' beta_cdr3: CDR3 of the beta chain only \n'+
                               ' \n defaults to `full`',
                        dest = 'model_type'
                       )
    parser.add_argument('-folds',
                        type = int,
                        help = 'number of MCCV folds to perform, default = 10',
                        dest = 'folds'
                       )  
    parser.add_argument('-multinomial_ids',
                        type = str,
                        help = 'either: `all`, `shared_by_all` or `shared_by_final`,'+
                               ' or ``public_key`. default=`shared_by_all` ',
                        dest = 'multinomial_ids')
    parser.add_argument('--randomize',
                       action='store_true',
                       help = 'Perform in non-reproducible way',
                       dest = 'randomize')
    
    
    args = parser.parse_args()
    
    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(0)
    
    if not args.data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
    
    if args.dir_out:
        output_path = args.dir_out
    else:
        output_path = None
        
    if args.mode:
        mode = args.mode
    else:
        mode = 'multinomial'
        
    if args.pmhc_do:
        pmhc_do = args.pmhc_do
    else:
        if mode == 'binomial':
            raise ValueError('No pmhc code was provided, and running in binomial mode')
        else:
            pmhc_do = None
    
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'full'
        
    if args.folds:
        folds = args.folds
    else:
        folds = 10
        
    if args.multinomial_ids:
        multinomial_ids = args.multinomial_ids
    else:
        multinomial_ids = 'shared_by_all'
        
    if args.randomize:
        seed = None
    else:
        seed = 0
    
    run_single_case(args.data_path,
                    output_path,
                    mode=mode,
                    pmhc_do = pmhc_do,
                    model_type = model_type,
                    folds = folds,
                    multinomial_ids = multinomial_ids,
                    seed = seed )
    
    
    
