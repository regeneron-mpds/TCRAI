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
import pandas as pd
import pathlib
import os
import json
import argparse
import sys
from scipy.optimize import fmin
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import numpy as np


def get_spec_sens_at_threshold(threshold,y_true,y_pred):
    y_bin = np.round( y_pred - (threshold-0.5) )
    tn, fp, fn, tp = confusion_matrix(y_true,y_bin).ravel()
    spec = tn/(tn+fp)
    sens = tp/(tp+fn)
    return spec,sens

def geo_mean_at_threshold(threshold,y_true,y_pred):
    spec,sens = get_spec_sens_at_threshold(threshold,y_true,y_pred)
    geo_mean = np.sqrt(spec*sens)
    return -geo_mean

def match_score_at_threshold(threshold,y_true,y_pred):
    spec,sens = get_spec_sens_at_threshold(threshold,y_true,y_pred)
    return (spec-sens)**2

def youden_score_at_threshold(threshold,y_true,y_pred):
    spec,sens = get_spec_sens_at_threshold(threshold,y_true,y_pred)
    return 1.0-sens-spec

def get_evals_dict(data_path, runs, keys):
    evals = {}
    for run,key in zip(runs, keys ):
        try:
            with open(os.path.join(data_path,run,'test_evals.json')) as f:
                tmp = json.load(f)
            evals[key] = tmp
        except:
            pass
    return evals

def apply_binomial_spec_sens(data_path, evals, runs, keys):
    for run,key in zip(runs, keys ):
        try:
            test_df = pd.read_csv(os.path.join(data_path,run,'test_results.csv'))
        except:
            continue
            
        thresholds = np.linspace(0.0001,0.999,400)
        scores = np.zeros_like(thresholds) 
        for i,thr in enumerate(thresholds):
            scores[i] = geo_mean_at_threshold(thr,test_df['labels'],test_df['preds'])

        best_thr_idx = np.argmin(scores)
        min_thr = thresholds[best_thr_idx]

        print(min_thr)
        spec,sens = get_spec_sens_at_threshold(min_thr,test_df['labels'],test_df['preds'])
        evals[key]['specificity'] = spec
        evals[key]['sensitivity'] = sens
            
    return None

def get_binomial_evals_df(data_path, runs, keys):
    evals = get_evals_dict(data_path, runs, keys)
    apply_binomial_spec_sens(data_path, evals, runs, keys)
    binomial_evals_df = pd.DataFrame.from_dict(data=evals,orient='index')
    return binomial_evals_df

def get_multinomial_evals_df(data_path, runs, keys):
    evals = get_evals_dict(data_path, runs, keys)
    multi_evals = pd.DataFrame.from_dict(data=evals,orient='index')
    return multi_evals


def save_results_dfs(data_path,ds,model_type,out_dir):
    path = pathlib.Path(data_path)
    runs = [i.name for i in path.glob('*/')]
    runs = [r for r in runs if 'unstable' not in r]

    binomial_runs = [r for r in runs if ('binomial' in r and 
                                         ds.split('.')[0] in r and
                                         model_type in r )]
    binomial_keys = [r.split('_')[-2] for r in binomial_runs]

    multi_runs = [r for r in runs if ('multinomial' in r and 
                                      ds.split('.')[0] in r and
                                      model_type in r)]
    multi_keys = [r.split('_')[-2] for r in multi_runs]

    bin_df = get_binomial_evals_df(data_path, binomial_runs, binomial_keys)
    mul_df = get_multinomial_evals_df(data_path, multi_runs, multi_keys)
    
    print(bin_df)
    print(mul_df)
    
    bin_df.to_csv(os.path.join(out_dir,ds+'_'+model_type+'_binomial_table.csv'))
    mul_df.to_csv(os.path.join(out_dir,ds+'_'+model_type+'_multinomial_table.csv'))
    return None

def save_fine_roc_curve_tables(data_path,ds,model_type,out_dir):
    path = pathlib.Path(data_path)
    runs = [i.name for i in path.glob('*/')]
    runs = [r for r in runs if 'unstable' not in r]

    binomial_runs = [r for r in runs if ('binomial' in r and 
                                         ds.split('.')[0] in r and
                                         model_type in r )]
    binomial_keys = [r.split('_')[-2] for r in binomial_runs]
    
    for run,key in zip(binomial_runs, binomial_keys ):
        try:
            test_df = pd.read_csv(os.path.join(data_path,run,'test_results.csv'))
        except:
            continue
        
        thresholds = np.linspace(0.0001,0.999,400)
        tprs = np.zeros_like(thresholds)
        fprs = np.zeros_like(thresholds)
        for i,thr in enumerate(thresholds):
            spec,sens = get_spec_sens_at_threshold(thr,test_df['labels'],test_df['preds'])
            fprs[i] = 1-sens
            tprs[i] = spec
        
        roc_df = pd.DataFrame({
            'threshold': thresholds,
            'tpr': tprs,
            'fpr': fprs
        })
        roc_df.to_csv(os.path.join(out_dir,ds+'_'+model_type+'_'+key+'_roc_table.csv'))
        
def save_tables(data_path,ds,model_type,do_roc,out_dir):
    save_results_dfs(data_path,ds,model_type,out_dir)
    if do_roc:
        roc_save_path = os.path.join(out_dir,'roc_data')
        if not os.path.exists(roc_save_path):
            os.makedirs(roc_save_path)
        save_fine_roc_curve_tables(data_path,ds,model_type,roc_save_path)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='path to all saved data',
                        )
    parser.add_argument('-o',
                        type=str,
                        help='output directory, default is current',
                        dest='out_dir'
                        )
    parser.add_argument('-ds',
                        type=str,
                        help='name of the dataset to get results for',
                        dest='ds'
                        )
    parser.add_argument('-model_type',
                        type=str,
                        help='name of the model type: full, alpha, beta etc',
                        dest='model_type'
                        )
    parser.add_argument('--roc',
                       action='store_true',
                       help='also save the fine-scale ROC table',
                       dest='do_roc')
    
    
    
    args = parser.parse_args()
    
    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(0)
    
    if not args.data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
    
    if args.ds:
        ds = args.ds
    else:
        ds = 'public_TCRs'
        
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.getcwd()
        
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'full'
        
    save_tables(args.data_path,ds,model_type,args.do_roc,out_dir)