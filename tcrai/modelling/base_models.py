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
""" Base models and model with processors for TCR problems

In order to have consistent models across the various problems one may want to solve, 
base models are defined.

BaseXCRModel provides a keras Model subclass, that is effectively abstract.

BaseXCRModelWithProcessors provides an abstract base class that contains a BaseXCRModel
subclass together with the processors used to process sequences into numerical representations,
and tokenizers used to process other inputs, such as the VJ genes. BY combining these, one can 
save the BaseXCRModelWithProcessors concrete subclass, and easily load from file, including 
the processors and tokenizers used in training. One can therefore simply pass the sequences 
and genes as text.


"""


import tensorflow as tf
import tensorflow.keras as keras
from abc import ABC, abstractmethod
import pickle
import os
import numpy as np

class BaseXCRModelWithProcessors(ABC):
    """ Base class for XCR Models, and associated processors
    
    Base class providing useful tools for handling multiple inputs and 
    passing those inputs into a BaseXCRModel. Evaluators can be added so that a single
    evaluation call can contain multiple results.
    Save and load functionality is provided allowing model and processors to be easily 
    saved and loaded together as a single object.
    
    Input can be passed as simple strings and gene names, rather than numerical representations.
    Typically, class methods expecting 'inputs' arguments, expect a dictionary with keys, 
    for example:
    {'ACDR3': array of strings,
     'BCDR3': array of strings,
     'V_beta': array of strings.......}
    
    
    parameters
    -----------
    model: (tcrai.modelling.base_models.BaseXCRModel subclass) 
            A BaseXCRModel subclass model.
    processors: (dict of tcrai.modelling.processors Processor) 
            A dict with form {name of expected model input : a processor object}
    extra_tokenizers: (dict of keras.tokenizers) 
            A dict with form  {name of expected model input : a keras tokenizer}
    load: (bool, optional, default=False)
            during loading of the model, optionally flag True for different behaviour.
            Useful in some subclasses.
    
    """
    
    def __init__(self,model,processors={},extra_tokenizers={},load=False):
        self.processors = processors
        self.model = model
        self.tokenizers = extra_tokenizers
        self.evaluators = dict()
        
    def add_evaluator(self, evaluator, name):
        """ add an evaluator with a name for later retrieval
        
        call evaluate() to see the results of the evaluation
        
        parameters
        ------------
        evaluator: (Callable) 
                callable taking inputs (y_true, y_pred)
        name: (string)
                A name for the evaluator
                    
        """
        self.evaluators[name] = evaluator
        
    def remove_evaluator(self,name):
        """ remove an evaluator
        
        parameters
        ------------
        name : (string)
                The name of the evaluator to be removed
        
        """
        del self.evaluators[name]
                
    def tokenize(self,inputs):
        """ Apply tokenizers to inputs
        
        parameters
        -----------
        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.
                
        returns
        ----------
        tokenized: (dict)
                dictionary of tokenized outputs for the inputs that correspond 
                to a tokenizer key of self.tokenizers. 
        
        """
        tokenized = dict()
        for tok_name,tok in self.tokenizers.items():
            tokenized[tok_name] = np.array(tok.texts_to_sequences(inputs[tok_name]))
        return tokenized
    
    def process_seqs(self,inputs):
        """ process input sequences
        
        process the inputs that correspond to named processors in self.processors.
        These are inputs that are CDRs.
        
        
         parameters
        -----------
        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.
                
        returns
        ----------
        pro_dict: (dict)
                dictionary of processed outputs for the inputs that correspond 
                to a processor key of self.processors.
                
        """
        pro_dict = dict()
        for k,v in self.processors.items():
            #pro_dict[k] = v.transform_seqs(inputs[k])
            pro_dict[k] = v.transform(inputs[k])['seqs']
        return pro_dict
    
    def unprocess(self,predictions):
        """ Unprocess predicted sequences
        
        From a sequence of integers representing a sequence as it would be 
        encoded by a processor of the model, revert the sequence into it's
        unprocessed (amino acid) form.
        
        predictions will be processed in place into sequences of chars.
        
        parameters
        -----------
        predictions: (dict)
                A dictionary of predictions of (integer) sequences. keys of predictions
                should match the keys of self.processors. values should be arrays of 
                lists of integers.
                
        returns
        ---------
        None
        
        """
        pred_seqs = dict()
        for k,v in self.processors.items():
            argmaxed = np.argmax(predictions[k],axis=-1).astype(int)
            predictions[k] = v.seqs_to_text( argmaxed )
        return None
    
    def process_pred_input(self,inputs):
        """ process inputs into format to be passed to keras model
        
        parameters
        ----------
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.
                
        returns
        ----------
        in_dict: (dict)
                dictionary of processed outputs for the inputs that correspond 
                to keys of self.processors and self.tokenizers.
        
        """
        pro_dict = self.process_seqs(inputs)
        tok_dict = self.tokenize(inputs)
        return {**pro_dict, **tok_dict}
    
    def simple_predict(self,inputs):
        """ make predictions from inputs
        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.
                
        returns
        ----------
        in_dict: (dict)
                dictionary of processed outputs for the inputs that correspond 
                to keys of self.processors and self.tokenizers.
        
        
        
        """
        pred_input = self.process_pred_input(inputs)
        preds = self.model.predict(pred_input)
        return preds
    
    @abstractmethod
    def run(self,inputs,**kwargs):
        """ make predictions using the internal model 
        
        abstract class method  - must be implemented in subclasses
        
        """
        pass
    
    def compile(self,**kwargs):
        """ compile the internal keras model
        
        parameters
        ------------
        
        **kwargs: keyword arguments as expected by keras model compile method.
        
        returns
        --------
        None
        
        """
        self.model.compile(**kwargs)
        
    @abstractmethod
    def fit(self,inputs,labels,**kwargs):
        """ Fit the internal model
        
        abstract method, subclasses must implement.
        
        parameters
        -----------
        inputs: (dict)
                dict of the input arrays.
                
        labels: (subclass dependent)
        
        **kwargs: 
             kwargs of the keras.Model fit method.
        
        """
        pass   
    
    def calculate_z(self,inputs,selection=None,first_seq=None):
        """ calculate the latent space / feature space ,z, of the model
        
        parameters
        -----------
        
        inputs: (dict)
                input dictionary from which to calculate_z
                
        selection: (list, optional, default=None)
                only those input names appearing in selection will be processed into
                z. If None, all inputs will be processed that appear in self.model's
                input list.
                
        first_seq: (string, optional, default=None):
                the sequence input with this key will appear as the first vector in z.
                If the key also appears in selection, the encoding will appear twice in
                z.
        
        returns
        ---------
        z: (np.array)
            numpy array of z. dimension (n_samples, (z_dim))
        
        
        """
        
        if selection is None:
            #selection=sorted(list(self.processors.keys())) + sorted(list(self.tokenizers.keys()))
            selection=self.model.input_list
        
        if not all( [ ((sel in self.processors) or (sel in self.tokenizers))  for sel in selection]):
            raise ValueError('Unknown key provided')
            
        in_dict = self.process_pred_input(inputs)
        
        z=[]
        
        if first_seq is not None:
            this_enc = self.model.seq_encoders[first_seq](in_dict[first_seq])
            if isinstance(this_enc,dict):
                x_e = this_enc['mu'].numpy()
                z.append(x_e)
            #selection = set(self.processors.keys())-set(first_seq)
                    
        
        for sel in selection:
            if sel in self.processors:
                this_enc = self.model.seq_encoders[sel](in_dict[sel])
                if isinstance(this_enc,dict):
                    x_e = this_enc['mu'].numpy()
                else:
                    x_e = this_enc.numpy()
                z.append(x_e)
            elif sel in self.tokenizers:
                z.append(self.model.extra_encoders[sel](in_dict[sel]).numpy())
            else:
                try:
                    z.append(self.model.extra_encoders[sel](in_dict[sel]).numpy())
                except:
                    raise RuntimeError("TCRAI ERROR: unknown input key passed to Model")
       
        return np.hstack(z)    
            
    def _apply_evaluations(self,predictions,labels):
        """ Calculate evaluation metrics as per internal evaluators
        
        parameters
        -----------
        predictions: (np.array)
                output predictions from the model
                
        labels: (np.array)
                Array of the labels
                
        returns 
        --------
        score_dict: (dict)
                Dictionary with keys as the names of the internal evaluators, and 
                values as the output of those evaluators
        
        """
        score_dict = dict()
        for name,e in self.evaluators.items():
            score_dict[name] = e(labels,predictions)
        
        return score_dict
    
    @abstractmethod
    def evaluate(self,inputs,labels,**kwargs):
        """ Evaluate model using internal evaluators
        
        Abstract, must be implemented in subclasses.
        
        parameters
        -----------
        inputs: (dict)
                dict of model inputs, to evaluate
                
        labels: (np.array)
                Array of the labels
                
        **kwargs: optional kwargs to pass to model.fit()
                
        returns 
        --------
        score_dict: (dict)
                Dictionary with keys as the names of the internal evaluators, and 
                values as the output of those evaluators
        
        
        """
        pass
    
    def _save_model(self,model_path):
        self.model.save(model_path)
        
    def save(self,directory):
        """ Save the model, and its processors and tokenizers
        
        The keras model, processors, tokenizers, and the internal model's input list
        will all be save into the given directory. Allowing for easy later loading of the 
        keras model, and the processors etc, so one doesn't have to worry about strong all
        the processors separtely.
        
        parameters
        ----------
        directory: (string, path)
                A string for the path to a directory in whcih to store the model. if the directory 
                doesn't exist, it will be created.
                
        returns 
        --------
        
        None
        
        
        """
        # pickle processor, save model
        pickle_path = os.path.join(directory, 'processors')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump( self.processors, open( os.path.join(pickle_path,'processors.pickle'), "wb" ) )
        model_path = os.path.join(directory, 'model')
        #self.model.save(model_path)
        self._save_model(model_path)
        pickle_path = os.path.join(directory, 'extra_tokenizers')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump( self.tokenizers, open( os.path.join(pickle_path,'tokenizers.pickle'), "wb" ) )
        pickle_path = os.path.join(directory, 'input_list')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump( self.model.input_list, open( os.path.join(pickle_path,'input_list.pickle'), "wb" ) )
        
           
    @classmethod
    def from_file(cls,directory,**kwargs):
        """ Load a model from file
        
        parameters
        -----------
        directory: (string, path)
                A string for the path to a directory in whcih to store the model. if the directory 
                doesn't exist, it will be created.
                
        ** kwargs: optional kwargs for keras.model.load_model
        
        returns
        ---------
        
        cls: An instantiation of the concrete subclass being loaded.
        
        """
        model_path = os.path.join(directory, 'model')
        pickle_path = os.path.join(directory, 'processors', 'processors.pickle')
        tok_path = os.path.join(directory, 'extra_tokenizers', 'tokenizers.pickle')
        input_path = os.path.join(directory, 'input_list', 'input_list.pickle')
        
        model = keras.models.load_model(model_path,**kwargs)
        processors = pickle.load( open( pickle_path, "rb" ) )
        toks = pickle.load( open( tok_path, "rb" ) )
        if os.path.exists(input_path):
            input_list = pickle.load( open(input_path, "rb") )
            model.input_list = input_list
        
        return cls(model,processors=processors,extra_tokenizers=toks,load=True)
    
    
class BaseXCRModel(keras.Model, ABC):
    """ Base model for XCR problems
    
    This tf.keras.Model subclass provides the underlying foundation for building 
    further subclassed models to solve XCR based problems.
    
    multiple sequences can be passed as input, as well as further inputs that are not 
    sequences. The base class provides functionality for the calculation of a feature 
    space, z, which is the features after the output from the sequence encoders and
    extra_encoders have been concatenated together. This _calculate_z method should be used
    in the call method of subclasses where possible for consistancy.
    
    The output from the encoders can be passed to the predictor, generating a final prediction.
    Note the predictor can be any keras model, and so the output dimension can be any dimension
    in principle.
    
    BaseXCRModel subclasses will typically be used in conjunction with a BaseXCRModelWithProcessors
    object. By creating a BaseXCRModelWithProcessors object with a BaseXCRModel as its model, along
    with processors for processing sequences, and tokenizers for tokenizing other inputs, a general
    model with extra functionality, including easy saving and loading can be created.
    
    parameters
    ----------
    
    seq_encoders: (dict)
            dictionary of sequence encoders : {'encoder name' : keras.Model }. The model will expect 
            the input data to be a dictionary with keys of sequences matching the keys of the required
            encoder.
            
    extra encoders: (dict)
            dictionary of encoders for extra variables: {'encoder name' : keras.Model }. 
            The model will expect the input data to be a dictionary with keys of extra variables 
            matching the keys of the required encoder. 
            
    predictor: (tf.keras.Model)
            A keras model that will take the features of all the encoded inputs, and return a prediction.
            
    input_list: (list, optional, default=None)
            A list of keys of the inputs that the model should actually process. If None, input list
            will default to an alphabetically sorted list of all keys in the seq_encoders and extra_encoders
            
    **kwargs: optional keyword arguments to be passed to the keras.Model superclass.
    
    
    """
    
    def __init__(self,
                 seq_encoders,
                 extra_encoders,
                 predictor,
                 input_list=None,
                 **kwargs
                ):
        super(BaseXCRModel,self).__init__(**kwargs)
        self.seq_encoders = seq_encoders
        self.extra_encoders = extra_encoders
        self.predictor = predictor
        if input_list is None:
            input_list = sorted(list(seq_encoders.keys()))+sorted(list(extra_encoders.keys()))
        self._set_input_list(input_list)
        self.z_concat = keras.layers.Concatenate(name='z_concat')
        
    def call(self,inputs,training=None):
        """ Used in fit/evaluate etc - subclasses should implement
        
        simple version would be:
        z = self._calculate_z(inputs)
        x = self.predictor(z)
        return x
        
        """
        pass
    
    def _set_input_list(self,input_list):
        self.input_list = input_list
        
    def _calculate_z(self,inputs,selection=None,training=None):
        """ calculate latent space features
        
        parameters
        ----------
        
        inputs: (dict)
                dict of (np/tf) arrays of the inputs, keys should be the names
                of the inputs, and values the arrays.
                
        selection: (list, optional, default=None)
                only those inputs whose keys are in selection will be processed into
                the feature space. If None, the self.input_list will be used
                
        returns
        -------
        
        z: (np or tf array)
            The feature space vector of the inputs appearing in selection_list.
        
        """
        if selection is None:
            selection=self.input_list
        
        vals = []
        for sel in selection:
            if sel in self.seq_encoders:
                try:
                    x_e = self.seq_encoders[sel](inputs[sel],training=training)
                except:
                    raise RunTimeError()
                if isinstance(x_e,dict):
                    x_e = x_e['mu']
                vals.append(x_e)
            elif sel in self.extra_encoders:
                vals.append(self.extra_encoders[sel](inputs[sel],training=training))
            else:
                raise RuntimeError("unknown selection key passed")
        
        if len(vals)>1:
            z = self.z_concat(vals)
        else:
            z = vals[0]
        return z
    
    