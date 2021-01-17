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
""" Module for underlying structure of of models for classication of XCR problems """
import tensorflow as tf
import tensorflow.keras as keras
from tcrai.modelling.base_models import BaseXCRModelWithProcessors,BaseXCRModel

class SeqClassificationModelWithProcessor(BaseXCRModelWithProcessors):
    """ model for sequences wrapped with processor to convert text to numeric input
    """
    
    def run(self,inputs):
        """ run the model on given input
        
        Model will be run in inference mode
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
        
        returns
        -------
        x: tf.Tensor
            Predictions of the model on each input sample.
        """
        return self.simple_predict(inputs)
    
    def fit(self,inputs,labels,**kwargs):
        """ Fit the model on given input
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
            
        labels: list-like
            list/array of labels for the inputs. Should have length the same as the values 
            in the input dictionary.
            
        kwargs: optional keyword arguments
            Kwargs should be those accepted by keras.Model.fit()
        
        returns
        --------
        history: tf.keras.History
            A tf.keras history object storing the fitting process results per epoch.
            
        """
        fit_input = self.process_pred_input(inputs)
        if 'validation_data' in kwargs:
            vali_in_dict = self.process_pred_input(kwargs['validation_data'][0])
            v_labels = kwargs['validation_data'][1]
            if len(kwargs['validation_data'])==3:
                v_sw = kwargs['validation_data'][2]
                vali_tuple = (
                    vali_in_dict,
                    v_labels,
                    v_sw
                )
            else:
                vali_tuple = (
                    vali_in_dict,
                    v_labels
                )
            kwargs.update({'validation_data': vali_tuple})
            
        return self.model.fit(fit_input,labels,**kwargs)
    
    def evaluate(self,inputs,labels,**kwargs):
        """ Evaluate model performance on given input
        
        Note that evaluating functions can be added to the [...]ModelWithProcessors object
        via the add_evaluator() method - seee BaseXCRModelWithProcessors class in modelling/base_models.
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
            
        labels: list-like
            list/array of labels for the inputs. Should have length the same as the values 
            in the input dictionary.
            
        kwargs: optional keyword arguments
            Kwargs should be those accepted by keras.Model.fit()
        
        returns
        --------
        score_dict: dict
            A dictionary of evaluations keyed by the names that were provided when
            evaluator functions were added to the [...]ModelWithProcessors object.
        """
        preds = self.run(inputs)
        score_dict = self._apply_evaluations(preds,labels)
        return score_dict

class VJXCRSeqModel(BaseXCRModel):
    """ Model for classifying X-cell-receptors with V/J and sequence info"""
    
    def call(self,inputs,training=None):
        """ call the model
        parameters
        -----------

        inputs: dict
            A dictionary with keys as the names of the encoders 
            provided on initialization. (see BaseXCR class for further info).
            Values should be arrays of the input data for that keyed input. 
            E.g you may have 'Vgene' as a key for one your inputs, and the values
            might be [3,5,7,2,9,7].

        training: bool, optional, default=None
            If False, treat as inference, if True, trat as training.
            Keras model fit()/evaluate()/test() will appropriately select. 

        returns
        -------
        x: tf.Tensor
            The model's prediction for all samples provided in inputs
        """
    
        z = self._calculate_z(inputs,training=training)
        x = self.predictor(z,training=training)
        return x
    


