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
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import numpy as np
    
def plot_roc_binomial(y_true,y_pred,label='',save_path=None):
    """ plot roc curve for binomial prediction
    
    parameters
    ----------
    
    y_true: list, np.array
        list of 0,1 corresponding to the true classes of the data
    
    y_pred: list,np.array 
        array of predictions
    
    label: string, optional, default = ''
        label for the ROC curve, defualt will not have a label except
        for the AUC-ROC value.
        
    save_path: string, optional, default=None
        if not None, path at which to save the figure. If None, will show,
        but not save the figure.
    
    """
    f,ax = plt.subplots(1,1,figsize=(8,8))

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr,'k', 
             label=label+" : roc={:.2f}".format(roc_auc) 
            )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(frameon=False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_roc_multinomial(y_true,y_pred, labels, save_path=None):
    """ plot roc curve for sparsely labelled multinomial predictions
    
    This plotting routine expects the labels to sparsely labelled, that is,
    integer labels, not a one-hot representation. The predictions however,
    should be  aprobability distribution over the labels.
    
    parameters
    ----------
    
    y_true: list, np.array
        list of integers corresponding to the true classes of the data
    
    y_pred: np.array , shape = [samples,n_classes]
        array of predictions
    
    labels: dict or list
        either a lit or dict of labels for the different classes. If a list,
        labels[i] should be the label for the i`th class in the sparse y_test 
        class. If a dict, keys should be integers follwoing the same logic.
        
    save_path: string, optional, default=None
        if not None, path at which to save the figure. If None, will show,
        but not save the figure.
    
    """
    binarizer = LabelBinarizer()
    y_test_oh = binarizer.fit_transform(y_true)
    n_classes = np.size(y_pred,1)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_oh[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    f,ax = plt.subplots(1,1,figsize=(8,8))

    cmap = plt.get_cmap('tab20')
    cols = cmap(np.arange(n_classes))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cols[i], 
                 label=labels[i]+": roc={:.2f}".format(roc_auc[i]) 
                )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(frameon=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_loss(history,save_path=None):
    """ Plot loss, and validation loss, from model history
    
    parameters
    -----------
    history: keras.Model history
        History from a tf.keras model training
    
    save_path: string, optional, default=None
        if not None, path at which to save the figure. If None, will show,
        but not save the figure.
    """
    f,(ax0) = plt.subplots(1,1,figsize=(10,7))

    ax0.plot(history.history['loss'], 'k')
    ax0.plot(history.history['val_loss'], 'r')

    ax0.set_xlabel('epochs')
    ax0.set_ylabel('loss')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()