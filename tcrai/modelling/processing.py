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
""" Module containing processors for sequences of amino acids

All processors should be a subclass of the AASeqProcessor object, conforming with the
methods therein.

"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Bio.Data.IUPACData import extended_protein_letters,protein_letters

class AASeqProcessor:
    """ A simple general processor for sequences of amino acids
    
    This processor will convert AA sequences (strings) into sequences
    of integers. Sequences will all be 0-padded to a fixed length which
    is chosen by the user.
    
    two forms of transformation are provided allowing for usage in different
    use cases. The transform_seqs method will do a simple transform of the 
    string inputs to integer sequence outputs. The transform method provides
    instead a dictionary output with the output sequences stored with key 'seqs'.
    This allows for other outputs, such as sample weights, or masks, to be provided 
    via the transform method in AASeqProcessor subclasses.
    
    parameters
    ------------
    max_len: (int)
            A maximum length of the output sequences - sequences will be padded with 
            0's such that alll sequences are the same length.
    
    extra_chars: (string, optional, default ='')
            A single string containing any characters that should be expected in the AA
            strings that will be passed as input. 
    
    """
    
    def __init__(self,max_len,extra_chars=''):
        self.max_len = max_len
        self.tokenizer = Tokenizer(char_level=True,lower=False,filters='',oov_token='U')
        self.undo_tokenizer = Tokenizer(char_level=True,lower=False,filters='')
        self._fit_tokenizer(self.tokenizer,protein_letters+extra_chars)
        self._fit_tokenizer(self.undo_tokenizer,'U'+protein_letters+extra_chars)
        
    def _fit_tokenizer(self,tokenizer,expected_chars):
        """ fit the internal tokenizer on expected characters (AAs)
        
        parameters
        ------------
        tokenizer: tensorflow.keras.preprocessing.text.Tokenizer
                The tokenizer to fit
                
        expected_chars: string
                String containing the characters that are expected in the texts
        
        """
        tokenizer.fit_on_texts([expected_chars])
        return None
    
    def pad(self,x):
        """ pad sequences to all be same length 
        
        parameters
        -----------
        x: list
            list of sequences to be padded
            
        returns
        --------
        padded_x: np.array
                numpy array of integer sequences of fixed length dim (len(x), self.max_len)
        """
        return pad_sequences(x,maxlen=self.max_len,padding='post')
    
    def seqs_to_basic_text(self,seqs):
        """ convert sequences back into text 
        
        parameters
        ------------
        seqs: list
            a list of sequences of integers to be converted to texts
            
        returns
        --------
        texts: list
            a list of strings of the converteds seqs
        
        """
        return self.undo_tokenizer.sequences_to_texts(seqs)
    
    def seqs_to_text(self,seqs):
        """ convert intger sequences back to text, removing padded spaces
        
        parameters
        -----------
        seqs: list
            a list of sequences of integers to be converted to texts
            
        returns
        --------
        out_txt: list
            a list of strings of the converteds seqs
        
        """
        to_return = self.seqs_to_basic_text(seqs)
        out_txt = []
        [out_txt.append(t.replace(" ", "")) for t in to_return]
        return out_txt  
    
    def transform_seqs(self,x):
        """ convert AA strings to integer array
        parameters
        -----------
        x: list
            a list of AA strings
            
        returns
        --------
        seqs: np.array
                numpy array of integer sequences of fixed length dim (len(x), self.max_len)
        
        """
        x = self.tokenizer.texts_to_sequences(x)
        seqs = self.pad(x)
        return seqs
    
    def transform(self,x,**kwargs):
        """ convert AA strings to integer array
        parameters
        -----------
        x: list
            a list of AA strings
            
        returns
        --------
        transformed: dict
            dict with at least the single key 'seqs', with values being converted AA sequences.
            Subclasses may provide further key-value pairs here.
        
        """
        return {'seqs': self.transform_seqs(x)}
    
    def _get_feature_dim(self):
        """ Returns dimension of the tokenizer, i.e number of characters it encodes """
        return len(self.tokenizer.word_index)+1
    
    @property
    def feature_dim(self):
        """ number of features at each seq position """
        return self._get_feature_dim()
    
    @property
    def vocab_size(self):
        """ number of tokens in tokenizer """
        return len(self.tokenizer.word_index)+1
