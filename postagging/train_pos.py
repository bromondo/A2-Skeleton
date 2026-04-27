import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

# Import our own files
from data.PoSData import Vocab, getUDPOSDataloaders
from models.PoSGRU import PoSGRU
import gensim.downloader

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs":256,   # batch size
    "lr":0.0005, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":30,
    "layers": 2,
    "embed_dim":100,
    "hidden_dim":256,
    "residual":True,
    "use_glove":True
}


def main():

  # Get dataloaders
  train_loader, val_loader, _, vocab = getUDPOSDataloaders(config["bs"])

  vocab_size = vocab.lenWords()
  label_size = vocab.lenLabels()

  
  ##################################
  #  Q11
  ##################################
  # Preload GloVE vectors
  if config['use_glove']:
    glove_wv = gensim.downloader.load('glove-wiki-gigaword-100')
    embed_init = torch.randn(vocab_size, 100)
    Wordsnotinlist = []
    #Create vocab_size x embed_size tensor to initialize embedding
    
    #Iterate through vocab words and copy over glove vectors when possible
    for i,w in vocab.idx2word.items():
      if w in glove_wv:
        embed_init[i] = torch.from_numpy(glove_wv[w].copy())
      else:
        Wordsnotinlist.append(w)
    print(f"number of words: {len(Wordsnotinlist)} of {vocab_size}")
    print("Examples:", Wordsnotinlist[:5])
    
  else:
    embed_init = None

  # Build model
  model = PoSGRU(vocab_size=vocab_size, 
                 embed_dim=config["embed_dim"], 
                 hidden_dim=config["hidden_dim"], 
                 num_layers=config["layers"],
                 output_dim=label_size,
                 residual=config["residual"],
                 embed_init=embed_init)
  print(model)


  # Start model training
  train(model, train_loader, val_loader)


############################################
# Skeleton Code
############################################

def train(model, train_loader, val_loader):

  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login(key="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")
  wandb.init(project="[AI539] UDPOS HW2", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  ##################################
  #  Q6
  ##################################
  # Set up optimizer and our learning rate schedulers
  warmup_epochs = int(.1 * config["max_epoch"])
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs])

  ##################################
  #  Q7 
  ##################################
  criterion = nn.CrossEntropyLoss(ignore_index=-1)

  # Main training loop with progress bar
  iteration = 0
  best_val_acc = 0.0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    for x, y, lens in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      

      ##################################
      #  Q7 / Q11
      ##################################
      loss = criterion(out.reshape(-1, out.shape[-1]),y.reshape(-1))

      loss.backward()
      grad_norm_words = model.embed.weight.grad.norm()
      optimizer.step()
      optimizer.zero_grad()


      
      nonpad = (y != -1).to(dtype=float).sum().item()
      acc = (torch.argmax(out, dim=2)==y).to(dtype=float).sum() / nonpad
      wandb.log({"Loss/train": loss.item(), "Acc/train": acc.item(), "Grad/norm":grad_norm_words.item()}, step=iteration)
      pbar.update(1)
      iteration+=1

    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)

    ##################################
    #  Q8
    ##################################
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), f"./chkpts/{run_name}_model.pt")
      with open(f"./chkpts/{run_name}_vocab.pkl", "wb") as f:
        pickle.dump(train_loader.dataset.vocab, f)
    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()

##################################
#  Q8
##################################
def evaluate(model, loader):
  model.eval()
  total_loss = 0.0
  acc = 0.0
  total_nonpad = 0.0
  criterion = nn.CrossEntropyLoss(ignore_index=-1)
  for x, y, lens in loader:
    x = x.to(device)
    y = y.to(device)
    out = model(x)
    loss = criterion(out.reshape(-1, out.shape[-1]),y.reshape(-1))
    nonpad = (y != -1).to(dtype=float).sum().item()
    acc += (torch.argmax(out, dim=2)==y).to(dtype=float).sum().item()
    total_loss    += loss.item() * nonpad
    total_nonpad  += nonpad
  val_loss = total_loss / total_nonpad
  val_acc  = acc / total_nonpad

  return val_loss, val_acc

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_UDPOS"
  return run_name



if __name__== "__main__":
    main()
