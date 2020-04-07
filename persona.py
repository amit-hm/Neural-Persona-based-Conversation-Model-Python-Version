from params import params
from data import data
from io import open
import string
import numpy as np
import pickle
import linecache
import math

import torch
import torch.nn as nn
from torch.autograd import Variable, backward


class attention_feed(nn.Module):
    
    def __init__(self,params):
        super(attention_feed, self).__init__()
        self.params=params
    
    #output: batch_size*dimension
    def forward(self,target_t,context,context_mask):
        context_mask_p=(context_mask-1)*100000000       #100 million    #batch_size*max_length_s
        atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)       
        #atten:batch_size*max_length_s (before sum() it is batch_size*max_length_s*1) ;context: batch_size*max_length_s*dimension; target_t: batch_size*dimension*1

        atten=atten+context_mask_p      #batch_size*max_length_s
        atten=nn.Softmax(dim=1)(atten)      #batch_size*max_length_s
        atten=atten.unsqueeze(1)        #batch_size*1*max_length_s
        context_combined=torch.bmm(atten,context).sum(1)        #context_combined: batch_size*dimension (before sum: batch_size*1*dimension)
        return context_combined

class softattention(nn.Module):
    
    def __init__(self,params):
        super(softattention, self).__init__()
        self.params=params
        self.attlinear1=nn.Linear(self.params.dimension,self.params.dimension,False)
        self.attlinear2=nn.Linear(self.params.dimension,self.params.dimension,False)
    
    #returns output of size batch_size*dimension
    def forward(self,target_t,context,context_mask):
        context_mask_p=(context_mask-1)*100000000
        atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)     
        #atten:batch_size*max_length_s (before sum() it is batch_size*max_length_s*1) ;context: batch_size*max_length_s*dimension; target_t: batch_size*dimension*1

        atten=atten+context_mask_p      #batch_size*max_length_s
        atten=nn.Softmax(dim=1)(atten)      
        atten=atten.unsqueeze(1)        #batch_size*1*max_length_s
        context_combined=torch.bmm(atten,context).sum(1)        #batch_size*dimension
        output1=self.attlinear1(context_combined)       #batch_size*dimension
        output2=self.attlinear2(target_t)       #batch_size*dimension
        output=nn.Tanh()(output1+output2)       #batch_size*dimension
        return output


class lstm_source_(nn.Module):
    
    def __init__(self,params):
        super(lstm_source_, self).__init__()     #to declare the class as a Torch Module class
        self.params=params
        self.sembedding=nn.Embedding(self.params.vocab_source,self.params.dimension,padding_idx=self.params.vocab_dummy) 
        #vocab_source=25010, dimension=512, vocab_dummy=25006
        self.sdropout=nn.Dropout(self.params.dropout)
        for num in range(1,self.params.layers*2+1):  #4 layers; 8 'slinear' attributes   of lstm_source_
            setattr(self,"slinear"+str(num),nn.Linear(self.params.dimension,4*self.params.dimension,False))

    #inputs is a list of 9 elements: 1st 8 tensors: batch_size*dimension; 9th: batch_size
    #output is a list of 8 elements: 4 'h's and 'c's, each 256*512
    def forward(self,inputs):
        outputs = []
        for ll in range(self.params.layers):        #loop 4 times
            prev_h=inputs[ll*2]         #first tensor of each loop while creating inputs in model_forward
            prev_c=inputs[ll*2+1]       #second tensor of each loop while creating inputs in model_forward
            
            # embedding the input vector i.e. the lines of size mini-batch
            if ll==0:
                x=self.sembedding(inputs[-1])       #x: batch_size*dimension
            else:
                x=outputs[ll*2-2]
                
            drop_x=self.sdropout(x)         #setting dropout on x
            drop_h=self.sdropout(inputs[ll*2])      #first tensor of each loop while creating inputs in model_forward, which consists of all zeros
            
            #input
            i2h=getattr(self,"slinear"+str(ll*2+1))(drop_x)     #batch_size*2048
            #previous hidden state
            h2h=getattr(self,"slinear"+str(ll*2+2))(drop_h)     #batch_size*2048
            
            #combining input and previous hidden state
            gates=i2h+h2h       #batch_size*2048
            
            # Input data(mini-batch) for each layer in 512 dimension
            reshaped_gates=gates.view(-1,4,self.params.dimension)       #256*4*512
                       
            #forget gate
            forget_gate= nn.Sigmoid()(reshaped_gates[:,2])      #256*512
            
            #input gate 
            in_gate= nn.Sigmoid()(reshaped_gates[:,0])      #256*512
            #candidate
            in_transform= nn.Tanh()(reshaped_gates[:,1])      #256*512
            
            #New candidate
            l1=forget_gate*inputs[ll*2+1]       #256*512    #prev_c: using second tensor of each loop while creating inputs in model_forward
            l2=in_gate*in_transform
            next_c=l1+l2        #256*512

            #output gate
            out_gate= nn.Sigmoid()(reshaped_gates[:,3])      #256*512
          
            #next hidden state
            next_h= out_gate*(nn.Tanh()(next_c))        #256*512
            
            #storing new candidate and next hidden state
            outputs.append(next_h)
            outputs.append(next_c)            
        return outputs
        

class lstm_target_(nn.Module):
    
    def __init__(self,params):
        super(lstm_target_, self).__init__()
        self.params=params
        self.embedding=nn.Embedding(self.params.vocab_target,self.params.dimension,padding_idx=self.params.vocab_dummy)
        #vocab_target=25010 ; vocab_dummy = 25006
        
        # creating speaker embeddings
        if self.params.PersonaMode:
            self.speaker_embedding=nn.Embedding(self.params.SpeakerNum,self.params.dimension)
            
        self.dropout=nn.Dropout(self.params.dropout)
        
        self.linear=nn.Linear(self.params.dimension,4*self.params.dimension,False)        
        if self.params.PersonaMode:
            self.linear_v=nn.Linear(self.params.dimension,4*self.params.dimension,False)
            
        for num in range(1,self.params.layers*2+1):     #4 layers; 8 'linear' attributes   of lstm_source_
            setattr(self,"linear"+str(num),nn.Linear(self.params.dimension,4*self.params.dimension,False))
            
        self.atten_feed=attention_feed(self.params)
        self.soft_atten=softattention(self.params)
    
    #inputs: 12 elements: 8 elements-output of lstm_source function; 1-context; 1-'t'th word; 1-padding; 1-Speaker ID
    def forward(self,inputs):
        context=inputs[self.params.layers*2]    #extracting context
        x_=inputs[self.params.layers*2+1]       #t-th word of each sentence; 256
        source_mask=inputs[self.params.layers*2+2]      #padding; 256*max_length_s
        outputs=[]
        
        for ll in range(self.params.layers):
            prev_h=inputs[ll*2]     #non-zero h
            prev_c=inputs[ll*2+1]     #non-zero c
            
            if ll==0:
                x=self.embedding(x_)
            else:
                x=outputs[ll*2-2]
                
            drop_x=self.dropout(x)
            drop_h=self.dropout(inputs[ll*2])
            
            #input, with dropout; batch_size*(dimension*4)
            i2h=getattr(self,"linear"+str(ll*2+1))(drop_x)
            #previous hidden state, with dropout; batch_size*(dimension*4)
            h2h=getattr(self,"linear"+str(ll*2+2))(drop_h)
            
            if ll==0:
                context1=self.atten_feed(inputs[self.params.layers*2-2],context,source_mask)
                #passing 3 elements: 1st- hidden state of last layer of last timestamp of LSTM source (batch_size*dimension); 
                                    #2nd- Context from LSTM source (batch_size*max_length_s*dimension);
                                    #3rd- padding (batch_size*max_length_s); 0 in the beginning, 1 in places where there are words
                #context1: batch_size*dimension
                drop_f=self.dropout(context1)
                f2h=self.linear(drop_f)
                
                gates=(i2h+h2h)+f2h         #batch_size*(dimension*4)
                
                if self.params.PersonaMode:
                    speaker_index=inputs[self.params.layers*2+3]
                    speaker_v=self.speaker_embedding(speaker_index)     #batch_size*dimension
                    speaker_v=self.dropout(speaker_v)
                    v=self.linear_v(speaker_v)      #batch_size*(dimension*4)
                    gates=gates+v       #batch_size*(dimension*4)
            else:
                gates=i2h+h2h
                
            reshaped_gates = gates.view(-1,4,self.params.dimension)
            
            #forget gate
            forget_gate=nn.Sigmoid()(reshaped_gates[:,2])
            
            #input gate
            in_gate=nn.Sigmoid()(reshaped_gates[:,0])
            #candidate
            in_transform= nn.Tanh()(reshaped_gates[:,1])
            
            #output gate
            out_gate=nn.Sigmoid()(reshaped_gates[:,3])
            
            #new candidate
            l1=forget_gate*inputs[ll*2+1]
            l2=in_gate*in_transform
            next_c=l1+l2
            
            #new hidden state
            next_h= out_gate*(nn.Tanh()(next_c))
            
            outputs.append(next_h)
            outputs.append(next_c)
            
        soft_vector=self.soft_atten(outputs[self.params.layers*2-2],context,source_mask)
        #elements being passed: hidden state of last layer of the current timestamp; context; source_padding
        #soft_vector: batch_size*dimension
        
        outputs.append(soft_vector)     #Output has 9 elements: 8 elements of current timestamp(4 h's, 4 c's); soft_vector
        return outputs


class softmax_(nn.Module):
    
    def __init__(self,params):
        super(softmax_, self).__init__()
        self.params=params
        self.softlinear=nn.Linear(self.params.dimension,self.params.vocab_target,False)
    
    #h: soft_vector(batch_size*dimension); y: truth labels(batch_size) of TARGET words
    def forward(self,h,y):
        h2y= self.softlinear(h)     #batch_size*vocab_target
        pred= nn.LogSoftmax(dim=1)(h2y)     #batch_size*vocab_target; values are between (-inf,0)
        
        if self.params.use_GPU:
            w=torch.ones(self.params.vocab_target).cuda()
        else:
            w=torch.ones(self.params.vocab_target)
        w[self.params.vocab_dummy]=0        #25006 is set to 0
        
        Criterion=nn.NLLLoss(w,size_average=False,ignore_index=self.params.vocab_dummy)
        #If the field size_average is set to False, the losses are instead summed for each minibatch
        #ignore_index (python:int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient
        #reduction is by deafult true i.e. 'mean': the sum of the output will be divided by the number of elements in the output
        
        err=Criterion(pred, y)
        return err,pred


class persona:
    
    def __init__(self, params):
        self.Data=data(params)
        self.params=params

        self.lstm_source =lstm_source_(self.params)
        self.lstm_target =lstm_target_(self.params)
        self.lstm_source.apply(self.weights_init)     # weights_init is a member function
        self.lstm_target.apply(self.weights_init)     # apply: Applies fn recursively to every submodule (as returned by .children()) as well as self.
       
        embed=list(self.lstm_source.parameters())[0]    #sembedding from lstm_source, 25010*512
        embed[self.params.vocab_dummy].data.fill_(0)    
        embed=list(self.lstm_target.parameters())[0]    #embedding from lstm_target, 25010*512
        embed[self.params.vocab_dummy].data.fill_(0)
        
        if self.params.use_GPU:
            self.lstm_source=self.lstm_source.cuda()
            self.lstm_target=self.lstm_target.cuda()
        self.softmax=softmax_(self.params)
        self.softmax.apply(self.weights_init)
        
        if self.params.use_GPU:
            self.softmax=self.softmax.cuda()
        self.output=self.params.output_file     # save/testing/log or save/testing/non_persona/log
        
        # Creating an output file if it doesn't exist
        if self.output!="":
            with open(self.output,"w") as selfoutput:
                selfoutput.write("")
                
        if self.params.PersonaMode:
            print("training in persona mode")
        else:
            print("training in non persona mode")
            
        self.ReadDict()
        self.read_dict()


    def weights_init(self,module):
        classname=module.__class__.__name__
        try:
            module.weight.data.uniform_(-self.params.init_weight,self.params.init_weight)    #init_weight=0.1
        except:
            pass
    
    #Builds a dictionary where key varies from 0 tolength of dictionary.txt file and values are unique words
    def ReadDict(self):
        self.dict={}
        dictionary=open(self.params.train_path+self.params.dictPath,"r").readlines()        #data/testing/vocabulary
        for index in range(len(dictionary)):        # Dictionary is 0-indexed
            line=dictionary[index].strip()
            self.dict[index]=line
    
    def read_dict():
        self.speaker_dict = {}
        dictionary=open(self.params.train_path+self.params.speakdictPath,"r").readlines()
        for index in range(len(dictionary)):
            line=dictionary[index].strip()
            self.speaker_dict[index] = line

    def IndexToWord(self,vector):
        if vector.dim()==0:
            vector=vector.view(1)
        string=""
        for i in range(vector.size(0)):
            try:
                string=string+self.dict[int(vector[i])]+" "
            except KeyError: 
                string=string+str(int(vector[i]))+" "
        string=string.strip()
        return string

    #it loops over each timestamps of "Word_s". It calls forward() of lstm_source_ in either train or eval mode.
    #if it's NOT in decoding mode, it loops over timestamps of "Word_t". It calls forward() of lstm_target_ in either train or eval mode.
    #the output of lstm_target is sent to softmax, to calculate error
    #if it's in training mode, .backward() is called
    def model_forward(self):
        self.context=Variable(torch.Tensor(self.Word_s.size(0),self.Word_s.size(1),self.params.dimension))      #256*maxlength_s*512
        #Word_s: batch_size*max_length
        
        if self.params.use_GPU:
            self.context=self.context.cuda()
            
        for t in range(self.Word_s.size(1)):        #maxlength_s
            inputs=[]
            if t==0:
                for ll in range(self.params.layers):        #layers=4
                    if self.params.use_GPU:
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension).cuda()))      #batch_size*dimension
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension).cuda()))      #batch_size*dimension
                    else:
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension)))     #batch_size*dimension
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension)))     #batch_size*dimension
            else:
                inputs=output
                
            inputs.append(self.Word_s[:,t])     #appending t-th column of Word_s i.e. t-th word of each of 256 sentences

            if self.mode=="train":
                self.lstm_source.train()        # Turn on the train mode
            else:
                self.lstm_source.eval()         # Turn on the evaluation mode
                
            output=self.lstm_source(inputs)         #forward() in lstm_source_ needs to be implemented  #list of 8 elements, each 256*512
                    
            if t==self.Word_s.size(1)-1:        #when t==maxlength_s-1
                self.last=output
                
            self.SourceVector=output[self.params.layers*2-2]        #256*512; last but one element of output: this is final hidden state of 4 layers
            self.context[:,t]=output[2*self.params.layers-2]        #256*512 at 't'th dimension of max_length
            
        if self.mode!="decoding":
            sum_err=0
            total_num=0
            for t in range(self.Word_t.size(1)-1):      #max_length_t-1
                lstm_input=[]
                
                if t==0:
                    lstm_input=output
                else:
                    lstm_input=output[:-1]
                    
                lstm_input.append(self.context)         #256*maxlength_s*512
                lstm_input.append(self.Word_t[:,t])     #t-th word of each sentence; 256
                lstm_input.append(self.Padding_s)       #256*max_length_s
                
                if self.params.PersonaMode:
                    lstm_input.append(self.SpeakerID)
                
                #lstm_input has 12 elements(including SpeakerID)
                    
                if self.mode=="train":
                    self.lstm_target.train()
                else:
                    self.lstm_target.eval()
                    
                output=self.lstm_target(lstm_input)
                #Output has 9 elements: 8 elements of current timestamp(4 h's, 4 c's); soft_vector: batch_size*dimension
                
                current_word=self.Word_t[:,t+1]         #(t+1)th timestamp of TARGET words
                err,pred=self.softmax(output[-1],current_word)
                sum_err=sum_err+err
                total_num=total_num+self.Left_t[t+1].size(0)        #adding the number of words in the timestamp of target sentence
            if self.mode=="train":
                sum_err.backward()      #Computes the sum of gradients of given tensors w.r.t. graph leaves.
            return sum_err.data, total_num

    #the function loops over batches of "open_train_file". If both max_length_s and max_length_t are <50, 
    #it calls model_forward(), which returns error
    def test(self):
        if self.mode=="test":
            open_train_file=self.params.train_path+self.params.dev_file     # data/testing/valid.txt
        sum_err_all=0 
        total_num_all=0
        End=0
        batch_n=0
        while End==0:
            End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseeID=self.Data.read_train(open_train_file,batch_n)
            # End: 1, if one of the line in the batch is empty
            # Word_s and Word_t: tensors, batch_size*max_length
            # Padding_s, Paddint_t: tensors, batch_size*max_length
            # Mask_s, Mask_t: dicts, max_length
            # Left_s, Left_t: dicts, max_length
            # AddresseeID: str
            # SpeakerID: tensor of 256 elements, each representing speaker ID number
            # Source: dict with tensors as its values, wrt addressee
            # Target: dict with tensors as its values, wrt speaker
            
            batch_n+=1
            if len(self.Word_s)==0 or End==1:
                break
            #if both max_length_s and max_length_t are less than 50
            if (self.Word_s.size(1)<self.params.source_max_length and self.Word_t.size(1)<self.params.target_max_length):
                self.mode="test"
                self.Word_s=Variable(self.Word_s)       #variable wrap Tensor, provides a method to perform backpropagation
                self.Word_t=Variable(self.Word_t)
                self.Padding_s=Variable(self.Padding_s)
                self.SpeakerID=Variable(self.SpeakerID)

                if self.params.use_GPU:
                    self.Word_s=self.Word_s.cuda()
                    self.Word_t=self.Word_t.cuda()
                    self.Padding_s=self.Padding_s.cuda()
                    self.SpeakerID=self.SpeakerID.cuda()
                    
                sum_err,total_num=self.model_forward()
                sum_err_all+=sum_err
                total_num_all+=total_num
        print("perp "+str((1/math.exp(-sum_err_all/total_num_all))))
        
        #printing perplexity value to log file
        if self.output!="":
            with open(self.output,"a") as selfoutput:
                selfoutput.write("standard perp "+str((1/math.exp(-sum_err_all/total_num_all)))+"\n")

    def update(self):
        lr=self.lr      #till iter 6 it's 1; after that keep halving it
        grad_norm=0
        
        for module in [self.lstm_source,self.lstm_target,self.softmax]:
            for m in list(module.parameters()):
                m.grad.data = m.grad.data*(1/self.Word_s.size(0))       #dividing by batch_size
                #tensor.data gives a tensor that shares the storage with tensor, but doesn't track history,
                #because weights have requires_grad=True, but we don't need to track this in autograd.
                
                grad_norm+=m.grad.data.norm()**2        
                #norm(): Returns the matrix norm or vector norm of a given tensor. 
                #Default: Frobenius() [square root of the sum of the absolute squares of its elements]
                
        grad_norm=grad_norm**0.5        #square-root
        
        if grad_norm>self.params.thres:     #thres=5
            lr=lr*self.params.thres/grad_norm
            
        for module in [self.lstm_source,self.lstm_target,self.softmax]:
            for f in module.parameters():
                f.data.sub_(f.grad.data * lr)   #inplace computation of (f.data-f.grad.data*lr)
    
    #It saves only parameters of the model
    def save(self):
        #syntax: torch.save(the_model.state_dict(), PATH) 
        #PATH: save/testing/modelx_xxxxx.pkl or save/testing/non_persona/modelx_xxxxx.pkl
        
        torch.save(self.lstm_source.state_dict(),self.params.save_prefix+str(self.iter)+"_source.pkl")
        torch.save(self.lstm_target.state_dict(),self.params.save_prefix+str(self.iter)+"_target.pkl")
        torch.save(self.softmax.state_dict(),self.params.save_prefix+str(self.iter)+"_softmax.pkl")
        print("finished saving")

    def saveParams(self):
        with open(self.params.save_params_file+".pickle","wb") as file:         # save/testing/params.pickle or save/testing/non_persona/params.pickle
            pickle.dump(self.params,file)

    #This function is called in decode_model class which loads the parameters which is saved in save()
    def readModel(self):
        self.lstm_source.load_state_dict(torch.load(self.params.model_file+"_source.pkl"))
        self.lstm_target.load_state_dict(torch.load(self.params.model_file+"_target.pkl"))
        self.softmax.load_state_dict(torch.load(self.params.model_file+"_softmax.pkl"))
        print("read model done")

    def clear(self):
        for module in [self.lstm_source,self.lstm_target,self.softmax]:
            module.zero_grad()

    def train(self):
        if self.params.saveModel:
            self.saveParams()
            
        if self.params.fine_tuning:
            target_model=torch.load(self.params.fine_tuning_model+"_target.pkl")
            linear_v=self.lstm_target.state_dict()["linear_v.weight"]
            speaker=self.lstm_target.state_dict()["speaker_embedding.weight"]
            target_model["linear_v.weight"]=linear_v
            target_model["speaker_embedding.weight"]=speaker
            self.lstm_source.load_state_dict(torch.load(self.params.fine_tuning_model+"_source.pkl"))
            self.lstm_target.load_state_dict(target_model)
            self.softmax.load_state_dict(torch.load(self.params.fine_tuning_model+"_softmax.pkl"))
            print("read model done")
            
        self.iter=0
        start_halving=False
        self.lr=self.params.alpha   # alpha=1
        print("iter  "+str(self.iter))
        self.mode="test"
        self.test()

        #till now the code was in "test" mode, input data was from valid.txt, logged into save/testing/log or save/testing/non_persona/log
        while True:
            self.iter+=1
            print("iter  "+str(self.iter))
            
            #printing iteration number to log file
            if self.output!="":
                with open(self.output,"a") as selfoutput:
                    selfoutput.write("iter  "+str(self.iter)+"\n")
                    
            if self.params.start_halve!=-1:         #start_halve=6
                if self.iter>self.params.start_halve:       #iter>6
                    start_halving=True
            
            if start_halving:
                self.lr=self.lr*0.5
                
            open_train_file=self.params.train_path+self.params.train_file       #data/testing/train.txt
            End=0
            batch_n=0
            while End==0:
                self.clear()        
                #Since the backward() function accumulates gradients, and we don’t want to mix up gradients between minibatches, 
                #we have to zero them out at the start of a new minibatch.
                
                End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseeID=self.Data.read_train(open_train_file,batch_n)
                # End: 1, if one of the line in the batch is empty
                # Word_s and Word_t: tensors, batch_size*max_length
                # Padding_s, Paddint_t: tensors, batch_size*max_length
                # Mask_s, Mask_t: dicts, max_length
                # Left_s, Left_t: dicts, max_length
                # AddresseeID: str
                # SpeakerID: tensor of 256 elements, each representing speaker ID number
                # Source: dict with tensors as its values, wrt addressee
                # Target: dict with tensors as its values, wrt speaker
                
                batch_n+=1
                if End==1:
                    break
                    
                train_this_batch=False
                
                #if max_length_s and max_length_t are both less than 60
                #if condition is false, it skips training the current batch
                if (self.Word_s.size(1)<60 and self.Word_t.size(1)<60):
                    train_this_batch=True
                    
                if train_this_batch:
                    self.mode="train"
                    self.Word_s=Variable(self.Word_s)
                    self.Word_t=Variable(self.Word_t)
                    self.Padding_s=Variable(self.Padding_s)
                    self.SpeakerID=Variable(self.SpeakerID)
                    
                    if self.params.use_GPU:
                        self.Word_s=self.Word_s.cuda()
                        self.Word_t=self.Word_t.cuda()
                        self.Padding_s=self.Padding_s.cuda()
                        self.SpeakerID=self.SpeakerID.cuda()
                        
                    self.model_forward()
                    self.update()
            self.mode="test"
            self.test()
            if self.params.saveModel:
                self.save()
            if self.iter==self.params.max_iter:
                break
