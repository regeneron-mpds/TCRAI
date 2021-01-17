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
from  sklearn.cluster import KMeans

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
from results_to_table import geo_mean_at_threshold
from infer_on_public import MODEL_TYPE_COLS
from stacked_binomial_plot import PUBLIC_NAMES,INTERNAL_NAMES

pd.options.mode.chained_assignment = None

HOME = os.path.expanduser('~')

NAMES = {
    'public_TCRs' : 'Public',
    'CNN-prediction-with-REGN-pilot-version-2' : '10x',
}

COLS = ['darkorange',
       'seagreen',
       'gold',
       'blueviolet',
       'deeppink',
       'indianred',
       'yellow',
       'lawngreen']

def set_rc_params(figsize = (3.6,2.9)):
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['font.weight'] = 'normal'
    mpl.rcParams['font.size'] = 7.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['legend.fontsize'] = 'small' #'x-small'
    mpl.rcParams['figure.figsize'] = figsize
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    mpl.rcParams['xtick.major.pad'] = 1.0
    mpl.rcParams['ytick.major.pad'] = 1.0
    mpl.rcParams['axes.labelpad'] = 1.0
    return None

def make_subdir(path,subdir):
    new_path = os.path.join(path, subdir)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

def df_to_input(df):
    cols = ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRA_cdr3']
    x = { c:df[c].values for c in cols}
    return x

def save_cluster_fasta(df, ds, pmhc, c_idx, outpath, suffix='', chain='TRB_cdr3'):
    
    cdr_df = df[df['cluster']==c_idx][['TCR_id',chain]]

    outfile = os.path.join(outpath,pmhc+'_cluster'+'{:01d}'.format(c_idx)+'_'+chain+'_'+suffix+'.fasta')
    print("saving fasta to ", outfile)
    motif_extraction.cdr3s_to_fasta(cdr_df['TCR_id'].map(lambda x: str(x)).values,
                                    cdr_df[chain].values,
                                    outfile)
    return outfile

def save_cluster_vj(df,ds,pmhc,c_idx,outpath,suffix='',gene='TRB_v_gene'):
    
    cdr_df = df[df['cluster']==c_idx][['TCR_id',gene]]

    counts_series = cdr_df.groupby(gene).count()['TCR_id']

    set_rc_params(figsize=(1.0,1.0))

    f,ax = plt.subplots(1,1)
    ax.pie(counts_series.values) #, labels=counts_series.index)

    plt.tight_layout()
    plt.savefig(os.path.join(outpath,pmhc+'_cluster'+'{:01d}'.format(c_idx)+'_pie_'+gene+'.pdf'))
    
    counts_series = counts_series.sort_values(ascending=False)
    counts_series.to_csv(os.path.join(outpath,pmhc+'_cluster'+'{:01d}'.format(c_idx)+'_pie_'+gene+'_table.csv'))
    return None

def save_cluster_df(df,ds,pmhc,c_idx,outpath):
    cdr_df = df[df['cluster']==c_idx]
    cdr_path = os.path.join(outpath,pmhc+'_df_cluster'+'{:01d}'.format(c_idx)+'.csv')
    cdr_df.to_csv(os.path.join(cdr_path))
    return None

def plot_umap(z_train_umap,
              train_df,
              ds,
              pmhc,
              fig_path):
    
    set_rc_params()
    
    if ds=='public_TCRs':
        bkg_col = 'b'
    else:
        bkg_col = 'k'

    f,ax = plt.subplots(1,1)

    ax.scatter(z_train_umap[train_df.cluster.isna(),0],
                z_train_umap[train_df.cluster.isna(),1],
                c=bkg_col,
                s=1,
                alpha=0.1,
                label = 'nonbinder or weak prediction'
            )
    for c_idx in range(int(train_df['cluster'].max()+1)):
        this_df = train_df[train_df['cluster']==c_idx]
        if len( this_df )>9:
            ax.scatter(z_train_umap[train_df['cluster'].values==c_idx,0],
                        z_train_umap[train_df['cluster'].values==c_idx,1],
                        c=COLS[c_idx],
                        s=3,
                        alpha=0.5,
                        label = 'cluster '+str(c_idx)
                    )
        else:
            ax.scatter(z_train_umap[train_df['cluster'].values==c_idx,0],
                        z_train_umap[train_df['cluster'].values==c_idx,1],
                        c=COLS[c_idx],
                        s=1,
                        alpha=0.1
                    )
    
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.xticks([])
    plt.yticks([])

    plt.legend()

    if 'public' in ds:
        plt.title(PUBLIC_NAMES[pmhc]+'\n'+NAMES[ds]+' data')
    else:
        plt.title(INTERNAL_NAMES[pmhc]+'\n'+NAMES[ds]+' data')
        
    plt.tight_layout()

    plt.savefig(fig_path)
    ax.get_legend().remove()
    plt.savefig(fig_path[:-4]+'_nolegend.pdf')
    return None

def apply_optimal_clustering(z,df,threshold=0.5):
    
    #df_binders = train_df[train_df['binds']==1]
    z_binders = z[(df['binds']==1) & (df['preds']>threshold)]
    tcr_ids = df[(df['binds']==1) & (df['preds']>threshold)]['TCR_id']
    
    #df_nonbinders = train_df[train_df['binds']==0]
    #z_nonbinders = z_train_umap[train_df['binds']==0]
    
    labels = np.zeros((len(z_binders,)))
    best_score = -2
    for n in range(2,8):
        kmeans = KMeans(n_clusters= n,
                       random_state=42)
        tmp_labels = kmeans.fit_predict(z_binders)
        score = metrics.silhouette_score(z_binders,tmp_labels)
        if score>best_score:
            best_score = score
            labels = tmp_labels
    
    print("selected ", np.amax(labels)+1, " clusters")
    label_df = pd.DataFrame({'TCR_id': tcr_ids, "cluster": labels})
    df_out = df.join(label_df.set_index('TCR_id'),on='TCR_id')
    return df_out
    
    
def select_threshold(df):
    thresholds = np.linspace(0.01,0.99,300)
    scores = np.ones_like(thresholds)
    for i,thr in enumerate(thresholds):
        scores[i] = geo_mean_at_threshold(thr,df['labels'],df['preds'])
    best_idx = np.argmin(scores)
    return thresholds[best_idx]
    
def run_clustering(data_path,
                   train_data_path,
                   ds,
                   model_type,
                   pmhc,
                   selection_type,
                   z_score_standardize,
                   save_dfs,
                   output_path):
    key = pmhc

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
        
    train_df['preds'] = model.run(df_to_input(train_df))
    
    #threshold = select_threshold(train_df)
    threshold = 0.95
    
    print("model threshold = ", threshold)
    
    umapper = dim_reduction.ScaledXCRUmap(model,
                                          df_to_input,
                                          scaler=scaler,
                                          selection=selection
                                         )
    
        
    z_train_umap = umapper.fit_transform(train_df)
    
    train_df_clustered = apply_optimal_clustering(z_train_umap,train_df,threshold=threshold)
    n_clusters = int(train_df_clustered['cluster'].max()+1)
    
    output_path = make_subdir(output_path,ds)
    output_path = make_subdir(output_path,pmhc)
    
    
    #fasta_path = make_subdir(output_path,'cluster_fasta')
    
    for c_idx in range(n_clusters):
        cluster_path = make_subdir(output_path,'cluster_{:01d}'.format(c_idx))
        
        fasta_path_beta = save_cluster_fasta(train_df_clustered, 
                                             ds, 
                                             pmhc, 
                                             c_idx, 
                                             cluster_path, 
                                             suffix='topbinders')
        fasta_path_alpha = save_cluster_fasta(train_df_clustered, 
                                             ds, 
                                             pmhc, 
                                             c_idx, 
                                             cluster_path, 
                                             suffix='topbinders',
                                             chain='TRA_cdr3')
        print( "logo output = ", os.path.join(fasta_path_beta.rsplit('.',1)[0]+'_logo.pdf'))
        logo.save_logo_from_fasta(fasta_path_beta,
                                  os.path.join(fasta_path_beta.rsplit('.',1)[0]+'_logo.pdf'),
                                  title = None,
                                  units='probability')
        logo.save_logo_from_fasta(fasta_path_alpha,
                                  os.path.join(fasta_path_alpha.rsplit('.',1)[0]+'_logo.pdf'),
                                  title = None,
                                  units='probability')

        #vj_path = make_subdir(output_path,'cluster_vj')
        for gene in ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene']:
            save_cluster_vj(train_df_clustered, ds, pmhc, c_idx,
                            cluster_path, 
                            suffix='topbinders',
                            gene=gene)

        if save_dfs:
            #df_outpath = make_subdir(output_path,'cluster_dfs')
            save_cluster_df(train_df_clustered,ds,pmhc,c_idx,cluster_path)
    
    fig_path = os.path.join(output_path,'cluster_kmeans_self_'+pmhc+'_'+selection_type+'_'+model_type+'.pdf')
    print("saving figure to : ", fig_path)
    plot_umap(z_train_umap,
              train_df_clustered,
              ds,
              pmhc,
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
                        help='name of the dataset to get results for',
                        dest='ds'
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
    parser.add_argument('--save_csvs',
                        action='store_true',
                        help='save clustered TCRs into csv',
                        dest='save_dfs'
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
        ds = 'public_TCRs.csv'
        
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
        
    run_clustering(args.data_path, 
                   args.train_data_path, 
                   ds, 
                   model_type, 
                   args.pmhc, 
                   selection_type,
                   args.standardize,
                   args.save_dfs,
                   out_dir)


