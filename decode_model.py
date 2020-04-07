from decode_params import decode_params
from data import data
from persona import *
from io import open
import string
import numpy as np
import pickle
import linecache
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, backward

class decode_model(persona):        #Inheriting from persona

    def __init__(self, params):
        #getting Params pickle file which was saved in persona.train()
        with open(params.decode_path+"/params.pickle", 'rb') as file:       
            model_params = pickle.load(file)
            
        #Copying those parameters from model_params which are not in params
        for key in model_params.__dict__:           
            if key not in params.__dict__:
                params.__dict__[key]=model_params.__dict__[key]
                
        self.params=params
        self.mode="decoding"
        
        if self.params.PersonaMode:
            print("decoding in persona mode")
        else:
            print("decoding in non persona mode")
            
        self.Data=data(self.params)     #Intializing EOT, EOS, beta and params 
        
        self.lstm_source =lstm_source_(self.params)
        self.lstm_target =lstm_target_(self.params)
        self.softmax =softmax_(self.params)
        if self.params.use_GPU:
            self.lstm_source=self.lstm_source.cuda()
            self.lstm_target=self.lstm_target.cuda()
            self.softmax=self.softmax.cuda()
            
        self.readModel()    #loading the model(only parameters) of first iteration in training
        self.ReadDict()     #buidling a dictionary of words, with keys from 0 to len(dictionary.txt)
        self.read_dict()
        
    def sample(self):
        self.model_forward()
        
        #setting max length of output: 1.5*max_length_s
        if self.params.max_length==0:
            #batch_max_dec_length=torch.ceil(1.5*self.Word_s.size(1))
            batch_max_dec_length=math.ceil(1.5*self.Word_s.size(1))         #Word_s.size(1) = max_length_s
        else:
            batch_max_dec_length=self.params.max_length
            
        completed_history={}
        if self.params.use_GPU:
            beamHistory=torch.ones(self.Word_s.size(0),batch_max_dec_length).long().cuda()      
            # batch_size*batch_max_dec_length, filled with ones
        else:
            beamHistory=torch.ones(self.Word_s.size(0),batch_max_dec_length).long()
            
        #for each timestamp
        for t in range(batch_max_dec_length):
            lstm_input=self.last        #8 elements: 4h's, 4c's of last timestamp of LSTM source run with TEST DATA
            lstm_input.append(self.context)
            
            if t==0:
                if self.params.use_GPU:
                    lstm_input.append(Variable(torch.LongTensor(self.Word_s.size(0)).fill_(self.Data.EOS).cuda()))
                    #size: batch_size, filled with 25008
                else:
                    lstm_input.append(torch.LongTensor(self.Word_s.size(0)).fill_(self.Data.EOS))
            else:
                lstm_input.append(Variable(beamHistory[:,t-1]))
                
            lstm_input.append(self.Padding_s)
            if self.params.PersonaMode:
                lstm_input.append(self.SpeakerID)
            
            #lstm_input has 11 elements (if PersonaMode is True)
            
            self.lstm_target.eval()     #setting eval mode
            output=self.lstm_target(lstm_input)     #output has 9 elements: 8 elements of current timestamp(4 h's, 4 c's); soft_vector
            self.last=output[:-1]   #8 elements
            
            #pred: batch_size*vocab_target; values are between (-inf,0)
            if self.params.use_GPU:
                err,pred=self.softmax(output[-1],Variable(torch.LongTensor(output[-1].size(0)).fill_(1).cuda()))
                # 2nd part of softmax does not matter, as pred is based on only output[-1] i.e. soft_vector
            else:
                err,pred=self.softmax(output[-1],torch.LongTensor(output[-1].size(0)).fill_(1))
                
            prob=pred.data
            prob=torch.exp(prob)        #batch_size*vocab_target
            
            #if unknown words are not allowed, 1st column is filled with zero
            if not self.params.allowUNK:    
                prob[:,0].fill_(0)
                
            #when 'StochasticGreedy' = 1, we choose the word with highest probability for each sample
            #next_words has, for each sample, the index of the word with highest probability
            if self.params.setting=="StochasticGreedy":
                select_P,select_words=torch.topk(prob,self.params.StochasticGreedyNum,1,True,True)      #both: batch_size*k
                #topk: Returns the k largest elements of the given input tensor along a given dimension. (here k=1)
                #A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor
                prob=F.normalize(select_P, 1, dim=1)        #batch_size*k
                #Performs Lp normalization of inputs over specified dimension. dim=1: Euclidean norm
                next_words_index=torch.multinomial(prob, 1)     #batch_size*1
                #Returns a tensor where each row contains 'num_samples' indices sampled from the 
                #multinomial probability distribution located in the corresponding row of tensor input
                
                if self.params.use_GPU:
                    next_words=torch.Tensor(self.Word_s.size(0),1).fill_(0).cuda()      #batch_size*1
                else:
                    next_words=torch.Tensor(self.Word_s.size(0),1).fill_(0)
                    
                for i in range(self.Word_s.size(0)):        #batch_size
                    next_words[i][0]=select_words[i][next_words_index[i][0]]
            elif self.params.setting=="sample":
                next_words=torch.multinomial(prob, 1)
            else:
                next_words=torch.max(prob,dim=1)[1]
                
            #assigning 1 for those samples which has reached END
            end_boolean_index=torch.eq(next_words,self.Data.EOT)    #batch_size*1
            #Computes element-wise equality
            #The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
            
            if self.params.use_GPU:
                end_boolean_index=end_boolean_index.cuda()
                
            #if atleast one of the sample has reached END
            if end_boolean_index.sum()!=0:
                for i in range(end_boolean_index.size(0)):      #batch_size
                    if end_boolean_index[i][0]==1:      
                        example_index=i     #index of the sample which has reached END
                        if example_index not in completed_history:
                            if t!=0:
                                completed_history[example_index]=beamHistory[example_index,:t]      #storing the predicted sentence
                            else:
                                if self.params.use_GPU:
                                    completed_history[example_index]=torch.Tensor(1,1).fill_(0).cuda()      
                                else:
                                    completed_history[example_index]=torch.Tensor(1,1).fill_(0)
                                    
            beamHistory[:,t]=next_words.view(-1)
            
        #for samples which could not finish within 'batch_max_dec_length', just take first 'batch_max_dec_length' predicted words
        for i in range(self.Word_s.size(0)):        #batch_size
            if i not in completed_history:
                completed_history[i]=beamHistory[i,:]
        return completed_history


    def decode(self):
        open_train_file=self.params.train_path+self.params.DecodeFile       #data/testing/test.txt
        speaker_id = self.params.SpeakerID
        if not self.params.PersonaMode:
            speaker_id=0
        output_file=self.params.OutputFolder+"/"+self.params.train_path.split("/")[-1]+"_s"+str(speaker_id)+"_"+self.params.DecodeFile[1:]
        #outputs/testing_s2_test.txt
        
        with open(output_file,"w") as open_write_file:
            open_write_file.write("")
        End=0
        batch_n=0
        n_decode_instance=0
        
        while End==0:
            End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseID=self.Data.read_train(open_train_file,batch_n)
            if len(self.Word_s)==0:
                break
            n_decode_instance=n_decode_instance+self.Word_s.size(0)     #adding batch_size
            
            if self.params.max_decoded_num!=0 and n_decode_instance>self.params.max_decoded_num:
                break
                
            batch_n=batch_n+1
            self.mode="decoding"
            self.SpeakerID.fill_(self.params.SpeakerID-1)   #SpkeakerID of size batch_size, filled with (SpeakerID-1)
            
            self.Word_s=Variable(self.Word_s)
            self.Padding_s=Variable(self.Padding_s)
            self.SpeakerID=Variable(self.SpeakerID)
            if self.params.use_GPU:
                self.Word_s=self.Word_s.cuda()
                self.Padding_s=self.Padding_s.cuda()
                self.SpeakerID=self.SpeakerID.cuda()
                
            completed_history=self.sample()
            self.OutPut(output_file,completed_history)
            if End==1:
                break
        print("decoding done")

    def OutPut(self,output_file,completed_history):
        for i in range(self.Word_s.size(0)):    #batch_size
            if self.params.output_source_target_side_by_side:
                print_string=self.IndexToWord(self.Source[i].view(-1))      #source
                print_string=print_string+"|"
                print_string=print_string+self.IndexToWord(completed_history[i].view(-1))       #target
                with open(output_file,"a") as file:
                    file.write(print_string+"\n")
            else:
                print_string=self.IndexToWord(completed_history[i].view(-1))
                with open(output_file,"a") as file:
                    file.write(print_string+"\n")

