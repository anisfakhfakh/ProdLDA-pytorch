import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import log_normal

class ProdLDA(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_topics = num_topics
        self.dropout = nn.Dropout(dropout)
        self.encLayer1 = nn.Linear(self.vocab_size, self.hidden_size)
        self.encLayer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.LDAposterior = log_normal.LogNormal

        self.inferenceLayer_mu  = nn.Linear(self.hidden_size, self.num_topics)
        self.inferenceLayer_std = nn.Linear(self.hidden_size, self.num_topics)
        self.inferenceLayerbn_mu = nn.BatchNorm1d(self.num_topics)
        self.inferenceLayerbn_std = nn.BatchNorm1d(self.num_topics)

        self.decLayer = nn.Linear(self.num_topics, self.vocab_size)
        self.decbn =nn.BatchNorm1d(self.vocab_size)


    def encoder(self, inputs):
        '''
        Encoder part of the model
        inputs: documents (tokenized text)
        '''
        hidden = F.softplus(self.encLayer1(inputs))
        hidden = F.softplus(self.encLayer2(hidden))
        hidden = self.dropout(hidden)
        return(hidden)

    def decoder(self, inputs):
        '''
        Devceder part of the model
        inputs: sampled vectors from posterior
        '''
        inputs = self.dropout(inputs)
        outputs = self.decLayer(inputs)
        outputs = F.softmax(self.decbn(outputs), dim = 1)
        return(outputs)

    
    def forward(self, inputs, training):
        '''
        inputs: documents (tokenized text)
        training: boolean (True: training phase and False: evaluation phase)
        '''
        hidden = self.encoder(inputs) #encode to hidden dimension
        mu_hidden  = self.inferenceLayerbn_mu(self.inferenceLayer_mu(hidden))  #f_mu
        std_hidden = self.inferenceLayerbn_std(self.inferenceLayer_std(hidden)) #f_sigma
        posterior = self.LDAposterior(mu_hidden, (0.5 * std_hidden).exp()) #distribution q
        if training==True:
            sampled_hidden = posterior.rsample() #infer during training
        else:
            sampled_hidden = posterior.mean #use mean and variance during evaluation
        rec_output = self.decoder(sampled_hidden) #regenreate word dist from sampled
        return posterior, rec_output
