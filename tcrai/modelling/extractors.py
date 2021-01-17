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
""" Module for feature extractors - conversion of input into numerical vectors 

A series of functions that return keras Models, turning input data into a feature vector.

All functions should accept 'hp' as the first argument - a dictionary of hyperparameters.

Models should have an output shape of [batch, features] ideally.

"""
import tensorflow as tf
import tensorflow.keras as keras


def conv_seq_extractor(hp,seq_len,vocab_size,name=None):
    """ convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------
    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    
    """
    model_in = keras.Input(shape = (seq_len,))
    
    embedder = keras.layers.Embedding(vocab_size,hp['embed_dim'])
    
    convs=[]
    bns = []
    
    strides = [1]*len(hp['filters'])
    if hp['strides']:
        strides=hp['strides']
    
    for f,k,d,s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
        convs.append( keras.layers.Conv1D(f,
                                  kernel_size=k,
                                  dilation_rate=d,
                                  strides=s,
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                  padding='same')
                    )
        bns.append( keras.layers.BatchNormalization() )
                    
    
    elu_activation = keras.layers.ELU()
    
    dropout_conv = keras.layers.Dropout(hp['dropout_conv'])
    
    x = embedder(model_in)
    
    for c,bn in zip(convs,bns):
        x = c(x)
        x = elu_activation(x)
        x = dropout_conv(x)
        x = bn(x)
        
    x = keras.layers.GlobalMaxPool1D()(x)
    
    return keras.Model(inputs=model_in, outputs=x, name=name)

def conv_seq_extractor_no_embed(hp,seq_len,feature_dim,name=None):
    """ convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------
    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    
    """
    model_in = keras.Input(shape = (seq_len,feature_dim))
    
    convs=[]
    bns = []
    
    strides = [1]*len(hp['filters'])
    if hp['strides']:
        strides=hp['strides']
    
    for f,k,d,s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
        convs.append( keras.layers.Conv1D(f,
                                  kernel_size=k,
                                  dilation_rate=d,
                                  strides=s,
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                  padding='same')
                    )
        bns.append( keras.layers.BatchNormalization() )
                    
    
    elu_activation = keras.layers.ELU()
    
    dropout_conv = keras.layers.Dropout(hp['dropout_conv'])
    
    x=model_in
    for c,bn in zip(convs,bns):
        x = c(x)
        x = elu_activation(x)
        x = dropout_conv(x)
        x = bn(x)
        
    x = keras.layers.GlobalMaxPool1D()(x)
    
    return keras.Model(inputs=model_in, outputs=x, name=name)

def vj_extractor(hp,name):
    """ Extract Gene info - embed and dropout
    
    parameters
    -----------
    hp: dict
        dictionary of hyperpareameters, keys:
         - vj_embed : int
             the dimension in which to embed the one-hot gene representation
         - vj_width: int
             the original dimension of the one-hot representation coming in
         - dropout: float
             how much dropout to apply after the embedding
             
    name: string
        name to give the extractor model
        
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    """
    vj_in = keras.Input(shape = (1,), name='vj_input')
    
    reshape_layer = keras.layers.Reshape([hp['vj_embed']])
    
    x_vj_mu = keras.layers.Embedding(hp['vj_width'],hp['vj_embed'])(vj_in)
    
    x_vj_mu = reshape_layer(x_vj_mu)
    
    x_vj_mu = keras.layers.Dropout(hp['dropout'])(x_vj_mu)

    return keras.Model(inputs=vj_in,outputs=x_vj_mu,name=name)


