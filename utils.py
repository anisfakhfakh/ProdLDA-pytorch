from torch.distributions import kl_divergence
import torch
from torch.distributions import log_normal
import numpy as np

def compute_loss(inputs, model, training):
    '''
    compute total loss L = kld and recontruction
    '''
    model = model
    inputs = inputs
    posterior_dist, rec_output = model(inputs, training)
    prior_dist = log_normal.LogNormal(torch.zeros_like(posterior_dist.loc), torch.ones_like(posterior_dist.scale)) #reference log normal distribution (0,1)
    rec_loss = - torch.sum(inputs*(rec_output+1e-10).log()) # 1+e-10 to prevent log(0) = nan
    KLD_loss = torch.sum( kl_divergence(posterior_dist, prior_dist))
    return rec_loss, KLD_loss

def train_step(model, optimizer, inputs):
    '''
    train model on 1 epoch, 1 batch
    '''
    model.train()
    optimizer.zero_grad()
    rec_loss, KLD_loss = compute_loss(inputs, model, True)
    loss = rec_loss + KLD_loss
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    return(loss, optimizer, model)

def train(model, optimizer, TrainData, ValData, epochs, batch_size):
    '''
    train process
    '''
    for epoch in range(epochs):
        # 1 epoch
        running_losses = [] #minibatch train losses list
        running_val_losses = [] #minibatch validation losses list
        epoch_indices = torch.randperm(TrainData.size(0)).split(batch_size)
        for _, idx in enumerate(epoch_indices , 1):
            # 1 batch train
            inputs = TrainData[idx]
            loss, optimizer, model = train_step(model, optimizer, inputs)
            running_losses.append(loss.item()/inputs.size(0))
        running_loss = sum(running_losses)/len(running_losses) #average train loss
        print("Train Epoch %d loss: %.4f"%(epoch, running_loss))
        epoch_val_indices = torch.randperm(ValData.size(0)).split(batch_size)
        for _, idx in enumerate(epoch_val_indices, 1):
            # 1 batch validation
            inputs = ValData[idx]
            loss = val_step(model, inputs)
            running_val_losses.append(loss.item()/inputs.size(0))
        running_val_loss = sum(running_val_losses)/len(running_val_losses)# average validation loss
        print("Validation Epoch %d loss: %.4f"%(epoch, running_val_loss))
    print('Finished Training')
    return(model)

def val_step(model, inputs):
    '''
    validate model on 1 epoch, 1 batch
    '''
    model.eval()
    rec_loss, KLD_loss = compute_loss (inputs, model, False)
    loss = rec_loss + KLD_loss
    loss = loss.mean()
    return(loss)

def data_transform(dir, vocab_size, dataset="train"):
    '''
    Load and Transform data from directory 
    '''
    data = np.load(dir+dataset+".txt.npy", encoding='bytes', allow_pickle=True)
    data = np.array([ np.bincount(x.astype('int'), minlength=vocab_size) for x in data if np.sum(x)>0 ])
    tensor = torch.from_numpy(data).float()
    return(tensor, data.shape[1])

    
def test(model, inputs):
    '''
    Test model (perplexity)
    '''
    rec_loss, _ = compute_loss (inputs, model, False)
    N = inputs.sum()
    perplexity = (rec_loss / N).exp()
    print("Perplexity = %.4f"%(perplexity))