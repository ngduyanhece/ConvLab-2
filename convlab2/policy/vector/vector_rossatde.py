import os
import json
import numpy as np
from convlab2.policy.vec import Vector
from transformers import AutoTokenizer, AutoModel
import torch 


class RossatdeVector(Vector):
    def __init__(self):
        """
        init the distillbert model 
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bandainamco-mirai/distilbert-base-japanese")
        self.model = AutoModel.from_pretrained("bandainamco-mirai/distilbert-base-japanese",return_dict=True)
        self.da_voc = ['x','i','e','td','f']
        self.generate_dict()
    
    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """
        self.act2vec = dict((a, i) for i, a in enumerate(self.da_voc))
        self.vec2act = dict((v, k) for k, v in self.act2vec.items())
        self.da_dim = len(self.da_voc)
        self.state_dim = 768

    def state_vectorize(self, state):
        tokens = self.tokenizer.tokenize(state['utter'])
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_ids)
        tokens_pt = torch.tensor([tokens_ids])
        outputs = self.model(tokens_pt)
        state_vec = torch.mean(outputs['last_hidden_state'],dim=1)
        return state_vec

    def action_devectorize(self,action_vec):
        act_idx = np.argmax(action_vec)
        return self.vec2act[act_idx]
        

    def action_vectorize(self,action):
        act_vec = np.zeros(self.da_dim)
        act_vec[self.act2vec[action]] = 1
        return act_vec


    