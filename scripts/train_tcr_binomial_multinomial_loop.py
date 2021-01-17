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
import itertools
import os
import sys
import argparse
import pandas as pd
from train_tcr_case import run_single_case

HOME = os.path.expanduser("~")
os.environ['PYTHONHASHSEED']=str(42)

MULTI_TYPES =['shared_by_all',
              'shared_by_final',
              'all']

def run_loop(data_path,
            output_path,
            model_type = 'full',
            folds = 5,
            seed = 42):
    
    data_root_basis = data_path.rsplit('.', 1)[0]
    id_df = pd.read_csv(os.path.join(data_root_basis+'_id_df.csv'))
    pmhcs = list(id_df.id)
    
    def do_binomial_multi_loop(model_choice):
        for pmhc in pmhcs:
            run_single_case(data_path,
                            output_path,
                            mode = 'binomial',
                            pmhc_do = pmhc,
                            model_type = model_choice,
                            folds = folds,
                            multinomial_ids = None,
                            seed = seed)

        for multi in MULTI_TYPES:
            run_single_case(data_path,
                            output_path,
                            mode = 'multinomial',
                            model_type = model_choice,
                            folds = folds,
                            multinomial_ids = multi,
                            seed = seed)
    
    if model_type=='loop_key_models':
        print("looping through full,alpha, and beta model types - this will take some time")
        for m in ['full','beta','alpha']:
            print("\n\n model type now = ",m,"\n")
            do_binomial_multi_loop(m) 
    else:
        print("doing model type", model_type)
        do_binomial_multi_loop(model_type) 
        
    return None
    
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
    parser.add_argument('-model_type',
                        type = str,
                        help = 'specific inputs only options are \n'+
                               ' full : V/J genes + CDR3 of alpha / beta chains \n'+
                               ' alpha : V/J genes + CDR3 of alpha chain only \n'+
                               ' beta : V/J genes + +CDR3 of beta chain only \n'+
                               ' seq : CDR3 of alpha and beta chains \n'+
                               ' gene : V/J genes of alpha and beta chains only \n'+
                               ' beta_cdr3: CDR3 of the beta chain only \n'+
                               ' loop_key_models: will loop through full, alpha, beta type models'
                               ' \n defaults to `full`',
                        dest = 'model_type'
                       )
    parser.add_argument('-folds',
                        type = int,
                        help = 'number of MCCV folds to perform, default = 5',
                        dest = 'folds'
                       )  
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
        
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'full'
        
    if args.folds:
        folds = args.folds
    else:
        folds = 5
        
    if args.randomize:
        seed = None
    else:
        seed = 42
        
    
    run_loop(args.data_path,
            output_path,
            model_type = model_type,
            folds = folds,
            seed=seed)