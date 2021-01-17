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
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import argparse
from sklearn.linear_model import LinearRegression
from motif_kmeans import set_rc_params

def plot_from_df(joint_df, figpath):
    
    x = np.log10(joint_df['public_TCRs'])
    y = joint_df['roc_full']-joint_df['roc_gene']

    lr = LinearRegression()
    lr.fit(np.expand_dims(x,1),y)
    
    fig,ax = plt.subplots(1,1)

    ax.set_xscale('log')

    ax.errorbar(joint_df['public_TCRs'],
                y,
                yerr = np.sqrt( joint_df['roc_err_full']**2 +
                                joint_df['roc_err_gene']**2
                              ),
                color='k',
                marker='o',
                markersize=4,
                linestyle='',
                fmt='',
                capsize=1.0
               )

    x_tmp = np.logspace(1.1, 3.3,100)
    y_tmp = lr.predict(np.expand_dims(np.log10(x_tmp),1))
    plt.semilogx(x_tmp,y_tmp,'k--',alpha=0.5)
    plt.plot([10**1.0,10**3.3],[0.0,0.0],'k')

    plt.xlim([10**1.1, 10**3.3])

    plt.xlabel('positive samples')
    plt.ylabel('AUC$_{full}$ - AUC$_{gene}$')

    plt.tight_layout()
    plt.savefig(figpath)
    return None

def add_errors_to_df(df,data_path,model_type='full'):
    xval_sigma = []
    xval_roc = []
    for pmhc in df.index:
        xval = pd.read_csv(os.path.join(data_path,
                                        'binomial_public_TCRs_'+pmhc+'_'+model_type,'cross_val_table.csv'),
                           header=[0,1],
                           index_col=0)
        xval_sigma.append( xval.iloc[xval['roc']['mu'].idxmax()].roc.sigma)
        xval_roc.append( xval.iloc[xval['roc']['mu'].idxmax()].roc.mu)
    df['xval_sigma'] = xval_sigma
    df['xval_roc'] = xval_roc
    df['roc_diff'] = np.abs( df['roc'] - df['xval_roc'] )
    df['roc_err'] = df[['xval_sigma','roc_diff']].max(axis=1)
    return df
    
def compare_gene_full(tables_path,
                     data_path,
                     gene_data_path,
                     outpath):
    gene_df = pd.read_csv(os.path.join(tables_path,'public_TCRs_gene_binomial_table.csv'),index_col=0)
    full_df = pd.read_csv(os.path.join(tables_path,'public_TCRs_full_binomial_table.csv'),index_col=0)
    counts_df = pd.read_csv('~/data/tcr/10x_paper_data/data_counts_summary.csv',index_col=0)
     
    gene_df = add_errors_to_df(gene_df,gene_data_path,model_type='gene')
    full_df = add_errors_to_df(full_df,data_path,model_type='full')
    
    joint_df = gene_df.join(full_df,lsuffix='_gene',rsuffix='_full')
    joint_df = joint_df.join(counts_df[['public_TCRs']])
    
    figpath = os.path.join(outpath,'fitted_gene_v_full_logx_err.pdf')
    plot_from_df(joint_df, figpath)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='path to all tabularized results',
                        )
    parser.add_argument('gene_data_path',
                        type=str,
                        help='path to all saved (post-testing/model) data for gene only models',
                        )
    parser.add_argument('tables_path',
                        type=str,
                        help='path to summary tables of results',
                        )
    parser.add_argument('-o',
                        type=str,
                        help='output directory, default is current',
                        dest='out_dir'
                        )
    
    args = parser.parse_args()
    
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)
    
    if not args.data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
        
    if not args.gene_data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
        
    if not args.train_data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
    
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.getcwd()
   
    compare_gene_full(args.tables_path,
                      args.data_path, 
                      args.gene_data_path,
                      out_dir)

