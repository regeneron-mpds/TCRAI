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
import argparse
import sys
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

from stacked_binomial_plot import PMHCS,PUBLIC_NAMES,INTERNAL_NAMES,COLORS,PUBLIC_SKIP


def make_multinomial_plot(data_path,ds,model_type,selection,ext,out_dir):
    run = os.path.join(data_path, 'multinomial_'+ds+'_'+selection+'_'+model_type)
    
    df = pd.read_csv(os.path.join(run,'test_results.csv') )
    
    n_classes =len( df['labels'].unique() )
    binarizer = LabelBinarizer()
    y_oh = binarizer.fit_transform(df['labels'].values)
    
    preds = np.array( df['preds'].map(lambda x: x.split('_')).to_list() ).astype(np.float)
    
    print(np.shape(preds) )
    print(np.shape(y_oh) )
    
    pickle_path = os.path.join(run,'int_pmhc_map.pickle')
    with open(pickle_path, "rb") as input_file:
        int_pmhc_map = pickle.load(input_file)
    
    print(int_pmhc_map)
    
    fprs = dict()
    tprs = dict()
    thrs = dict() 
    auc = dict()
    for i in range(n_classes):
        pmhc = int_pmhc_map[i]
        fprs[pmhc], tprs[pmhc], _ = metrics.roc_curve(y_oh[:, i], preds[:, i])
        auc[pmhc] = metrics.auc(fprs[pmhc], tprs[pmhc])

    print(auc)
    
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['font.weight'] = 'normal'
    mpl.rcParams['font.size'] = 7.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['legend.fontsize'] = 'x-small'
    mpl.rcParams['figure.figsize'] = (3.6,2.9)
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    mpl.rcParams['xtick.major.pad'] = 1.0
    mpl.rcParams['ytick.major.pad'] = 1.0
    mpl.rcParams['axes.labelpad'] = 1.0

    fig,ax = plt.subplots(1,1)

    for key in fprs.keys():
        if 'public' in ds:
            if key in PUBLIC_SKIP:
                continue
            label_key = PUBLIC_NAMES[key]
        else:
            label_key = INTERNAL_NAMES[key]
        ax.plot(fprs[key],tprs[key],
                color=COLORS[key],
                label=label_key+": {:.3f}".format(auc[key]))

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    ax.plot([0,1],[0,1],'k--')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.legend(
               loc = 'best'
#                loc='lower left',
#                ncol = 2,
#                bbox_to_anchor = (-0.01,1.01)
              )

    plt.tight_layout()
    
    print("saving figure to ", os.path.join(out_dir,ds+'_'+model_type+'_'+selection+'_multinomial'+ext))
    plt.savefig(os.path.join(out_dir,ds+'_'+model_type+'_'+selection+'_multinomial'+ext) )
    
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
    parser.add_argument('-selection',
                        type=str,
                        help='which pmhc selection used a runtime to plot:'+
                             '\n - `all`: all pmhcs'+
                             '\n - `shared_by_final` '+
                             '\n - `shared_by_all',
                        dest='selection'
                       )
    parser.add_argument('-ext',
                        type=str,
                        help='figure filetype extension',
                        dest='ext'
                        )
    
    
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
        ds = 'public_TCRs.csv'
        
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.getcwd()
        
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'full'
        
    if args.ext:
        ext = '.'+args.ext
    else:
        ext = '.png'
        
    if args.selection:
        selection=args.selection
    else:
        selection = 'all'
        
    make_multinomial_plot(args.data_path,
              ds,
              model_type,
              selection,
              ext,out_dir)

