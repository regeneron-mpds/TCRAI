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
import os
import sys
import argparse

import tensorflow as tf
import tensorflow.keras as keras

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import pickle
import pathlib

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl

from tcrai.modelling.classification import SeqClassificationModelWithProcessor
from tcrai.plotting import shapes
from tcrai.motif import motif_extraction,dim_reduction,logo
from infer_on_public import MODEL_TYPE_COLS
from stacked_binomial_plot import PUBLIC_NAMES,INTERNAL_NAMES

pd.options.mode.chained_assignment = None

HOME = os.path.expanduser('~')


# ellipse list for one ellipse is
# [x0,y0,a,b,theta] - a,b, are the x,y width penalties, and theta is rotation angle
# ELLIPSES = {
#     'NLVPMVATV': [
#         [-0.1, 6.0, 1.0, 1.0, 0.0],
#         [7.0, 6.0, 1.0, 1.0, 0.0],
#         [12.0, -0.8, 1.4, 1.4, 0.0],
#         [12.2, 3.4, 4.1, 1.2, 0.0],
#         [14.0, 8.0, 0.9, 0.9, 0.0]
#     ],
#     'ELAGIGILTV' : [
#         [0.1, 10.0, 1.2, 0.8, 35]
#     ],
#     'GILGFVFTL' : [
#         [-3.8, 6.0, 1.4, 1.4, 0.0],
#         [9.5, 2.7, 3.7, 2.3, 120.0]
#     ],
#     'GLCTLVAML' : [
#         [5.4, 6.3, 1.0, 1.0, 0.0],
#         [3.2, 7.2, 1.0, 1.0, 0.0],
#         [2.2, 10.5,1.3, 1.0, 10.0],
#         [9.3, 11.8, 0.5, 1.0,0.0]
#     ]
# }
# ELLIPSES = {
#     'NLVPMVATV': [],
#     'ELAGIGILTV' : [],
#     'GILGFVFTL' : [
#         [8.3,3.3,1.9,1.1,38.0]
#     ],
#     'GLCTLVAML' : [
#         [4.5,-1.3,2.5,2.5,0.0],
#         [-1.3,7.5,1.3,1.3,0.0]
#     ]  
# }
ELLIPSES = {
    'NLVPMVATV': [],
    'ELAGIGILTV' : [],
    'GILGFVFTL' : [],
    'GLCTLVAML' : []   
}

LABELS = {
    'public_TCRs':'public',
    'CNN-prediction-with-REGN-pilot-version-2': '10x'
}

def make_subdir(path,subdir):
    new_path = os.path.join(path, subdir)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

def set_rc_params(figsize = (3.6,2.9)):
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['font.weight'] = 'normal'
    mpl.rcParams['font.size'] = 7.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['legend.fontsize'] = 'x-small'
    mpl.rcParams['figure.figsize'] = figsize
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    mpl.rcParams['xtick.major.pad'] = 1.0
    mpl.rcParams['ytick.major.pad'] = 1.0
    mpl.rcParams['axes.labelpad'] = 1.0
    return None


def df_to_input(df):
    cols = ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRA_cdr3']
    x = { c:df[c].values for c in cols}
    return x

def save_cluster_vj(cdr_df,i,outpath,suffix='',gene='TRB_v_gene'):
    counts_series = cdr_df.groupby(gene).count()['TCR_id']
    set_rc_params(figsize=(1.0,1.0))
    
    f,ax = plt.subplots(1,1)
    ax.pie(counts_series.values) #, labels=counts_series.index)

    plt.tight_layout()
    plt.savefig(os.path.join(outpath,'ellipse'+'{:01d}'.format(i)+'_pie_'+gene+'.pdf'))
    
    counts_series = counts_series.sort_values(ascending=False)
    counts_series.to_csv(os.path.join(outpath,'ellipse'+'{:01d}'.format(i)+'_pie_'+gene+'_table.csv'))
    return None

def save_ellipse_fasta(z_umap, 
                       df, 
                       pmhc, 
                       ellipse_data,
                       outpath, 
                       suffix='', 
                       save_dfs=True, 
                       save_vj=True):
    
    for i,e in enumerate(ellipse_data):
        in_ellipse = motif_extraction.z_in_ellipse_bool(z_umap,*e)
        
        #cdr_df = df[['TCR_id','TRB_cdr3','binds']].iloc[in_ellipse]
        cdr_df = df.iloc[in_ellipse]
        cdr_df = cdr_df[cdr_df['binds']==1]
        
        if len(cdr_df)>0:
            outfile_beta = os.path.join(outpath,pmhc+'_cdr3beta_ellipse'+'{:02d}'.format(i)+'_'+suffix+'.fasta')
            print("saving fasta to ", outfile_beta)
            motif_extraction.cdr3s_to_fasta(cdr_df['TCR_id'].map(lambda x: str(x)).values,
                                            cdr_df['TRB_cdr3'].values,
                                            outfile_beta)
            outfile_alpha = os.path.join(outpath,pmhc+'_cdr3alpha_ellipse'+'{:02d}'.format(i)+'_'+suffix+'.fasta')
            motif_extraction.cdr3s_to_fasta(cdr_df['TCR_id'].map(lambda x: str(x)).values,
                                            cdr_df['TRA_cdr3'].values,
                                            outfile_alpha)
            if save_dfs:
                cdr_path = os.path.join(outpath,pmhc+'_df_ellipse'+'{:02d}'.format(i)+'_'+suffix+'.csv')
                cdr_df.to_csv(os.path.join(cdr_path))
            if save_vj:
                for gene in ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene']:
                    save_cluster_vj(cdr_df, i, outpath, suffix=suffix,gene=gene)

            logo.save_logo_from_fasta(outfile_beta,
                                      os.path.join(outfile_beta.rsplit('.',1)[0]+'_logo.pdf'),
                                      title = None,
                                      units='probability')
            logo.save_logo_from_fasta(outfile_alpha,
                                      os.path.join(outfile_alpha.rsplit('.',1)[0]+'_logo.pdf'),
                                      title = None,
                                      units='probability')
    
    return None

def plot_umap(z_train_umap,
              train_df,
              z_test_umap,
              test_df,
              pmhc,
              ds,
              infer_ds,
              fig_path):
    set_rc_params()

    if infer_ds=='public_TCRs':
        infer_cols = ['b','c']
        ds_cols = ['k','r']
    else:
        ds_cols = ['b','c']
        infer_cols = ['k','r']
        
    f,ax = plt.subplots(1,1)

    ax.scatter(z_train_umap[train_df['binds'].values==0,0],
                z_train_umap[train_df['binds'].values==0,1],
                c=ds_cols[0],
                s=1,
                alpha=0.1,
                label = LABELS[ds]+' - nonbinding'
            )
    ax.scatter(z_test_umap[test_df['binds'].values==0,0],
                z_test_umap[test_df['binds'].values==0,1],
                c=infer_cols[0],
                s=1,
                alpha=0.1,
                label = LABELS[infer_ds]+' - nonbinding'
            )
    ax.scatter(z_train_umap[train_df['binds'].values==1,0],
                z_train_umap[train_df['binds'].values==1,1],
                c=ds_cols[1],
                s=3,
                alpha=0.5,
                label = LABELS[ds]+' - binding'
            )
    ax.scatter(z_test_umap[test_df['binds'].values==1,0],
                z_test_umap[test_df['binds'].values==1,1],
                c=infer_cols[1],
                s=4,
                alpha=0.5,
                label = LABELS[infer_ds]+' - binding'
            )
    
    try:
        print("plotting ellipses")
        ellipse_data = ELLIPSES[pmhc]
        for i,e in enumerate(ellipse_data):
            shapes.plot_ellipse_on_ax(ax,*e)
            plt.text(e[0]+0.7*np.sqrt(e[2]),
                     e[1]+0.7*np.sqrt(e[3]),
                     '{:02d}'.format(i))
    except:
        print("ellipses plotting failed - no ellipses defined, or other error")
        pass
    
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.xticks([])
    plt.yticks([])

    plt.legend()

    
    plt.title(INTERNAL_NAMES[pmhc]+'\n '+LABELS[infer_ds] + ' inferred with ' + LABELS[ds] + ' model')

    plt.tight_layout()

    plt.savefig(fig_path)
    return None

def run_inference(data_path,
                   train_data_path,
                   ds,
                   infer_ds,
                   model_type,
                   pmhc,
                   selection_type,
                   z_score_standardize,
                   output_path):
    key = pmhc

    public_ds = infer_ds+'.csv'
    df_public_full = pd.read_csv(os.path.join(train_data_path,public_ds))

    train_ds = ds+'.csv'
    df_train_full = pd.read_csv(os.path.join(train_data_path,train_ds))
    
    path = pathlib.Path(data_path)
    mode = 'binomial'
    runs = [i.name for i in path.glob('*/')]
    runs = [r for r in runs if 'unstable' not in r]
    runs = [r for r in runs if (mode in r and 
                                ds in r 
                                and model_type in r
                                and key in r)]
    run = runs[0]

    model_path = os.path.join(data_path,run,'model')
    try:
        model = SeqClassificationModelWithProcessor.from_file(model_path)
    except:
        raise RuntimeError('no model found for '+ key + ' at\n' + model_path  )

    test_df = df_public_full[df_public_full['id']==key]
    test_df['labels'] = test_df['binds']
    test_df = test_df.drop_duplicates(subset=MODEL_TYPE_COLS[model_type]+['labels'])

    train_df = df_train_full[df_train_full['id']==key]
    train_df['labels'] = train_df['binds']
    train_df = train_df.drop_duplicates(subset=MODEL_TYPE_COLS[model_type]+['labels'])

    if selection_type=='all':
        selection = None
    elif selection_type == 'gene':
        selection = ['TRA_j_gene','TRA_v_gene','TRB_v_gene','TRB_j_gene']
    elif selection_type == 'cdr3s':
        selection = ['TRA_cdr3','TRB_cdr3']
    elif selection_type == 'beta_cdr3':
        selection = ['TRB_cdr3']
    else:
        raise ValueError("unnknown selection type: ", selection_type)
        
    if z_score_standardize:
        scaler = StandardScaler()
    else:
        scaler = None
        
    output_path = make_subdir(output_path,'inference')
    output_path = make_subdir(output_path,pmhc)
    output_path = make_subdir(output_path,'from_'+infer_ds+'_using_'+ds)
    print("output path = ", output_path)
    
    umapper = dim_reduction.ScaledXCRUmap(model,
                                          df_to_input,
                                          scaler=scaler,
                                          selection=selection
                                         )
    
       
    z_train_umap = umapper.fit_transform(train_df)
    z_test_umap = umapper.transform(test_df)
    
    
#     output_path = os.path.join(output_path, 'inference')
#     fasta_path = os.path.join(output_path, 'ellipse_fasta')
#     print("saving fasta to ... ", fasta_path)
#     if not os.path.exists(fasta_path):
#         os.makedirs(fasta_path)

    #fasta_path = make_subdir(output_path,'fasta')
    save_ellipse_fasta(z_test_umap, 
                       test_df, 
                       pmhc, 
                       ELLIPSES[pmhc],
                       output_path, 
                       suffix=infer_ds+'_binders')
    
    fig_path = os.path.join(output_path,
                            'cluster_'+pmhc+'_from'+infer_ds+'_using_'+
                            ds+'_'+selection_type+'_'+model_type+'.pdf')
    plot_umap(z_train_umap,
              train_df,
              z_test_umap,
              test_df,
              pmhc,
              ds,
              infer_ds,
              fig_path)
    return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        type=str,
                        help='path to all saved (post-testing/model) data',
                        )
    parser.add_argument('train_data_path',
                        type=str,
                        help='path to data used for training',
                        )
    parser.add_argument('-o',
                        type=str,
                        help='output directory, default is current',
                        dest='out_dir'
                        )
    parser.add_argument('-ds',
                        type=str,
                        help='name of the dataset trained model to use',
                        dest='ds'
                        )
    parser.add_argument('-infer_ds',
                        type=str,
                        help='name of the dataset to get results for',
                        dest='infer_ds'
                        )
    parser.add_argument('-model_type',
                        type=str,
                        help='name of the model type: full, alpha, beta etc',
                        dest='model_type'
                        )
    parser.add_argument('-pmhc',
                        type=str,
                        help='pmhc to study',
                        dest='pmhc'
                        )
    parser.add_argument('-select',
                        type=str,
                        help='the type of inputs to be selected \n:'+
                             ' \t all : all inputs' +
                             ' \t gene : just the genes' +
                             ' \t cdr3s : just the CDR3s' +
                             ' \t beta_cdr3 : just the beta chain_cdr3',
                        dest='selection_type'
                        )
    parser.add_argument('--standardize',
                        action='store_true',
                        help='apply z score norm to z',
                        dest='standardize'
                        )
    
    
    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    
    if not args.data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
        
    if not args.train_data_path:
        print("ERROR: No data path was passed")
        sys.exit(0)
    
    if args.ds:
        ds = args.ds
    else:
        ds = 'public_TCRs'
        
    if args.infer_ds:
        if args.infer_ds[-3:]=='csv':
            infer_ds = args.infer_ds.split('.')[0]
        else:
            infer_ds = args.infer_ds
    else:
        infer_ds = 'public_TCRs'
        
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.getcwd()
        
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'full'
        
    if args.selection_type:
        selection_type = args.selection_type
    else:
        selection_type = 'all'
        
    run_inference(args.data_path, 
                   args.train_data_path, 
                   ds, 
                   infer_ds,
                   model_type, 
                   args.pmhc, 
                   selection_type,
                   args.standardize,
                   out_dir)


