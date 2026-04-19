import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from data.BPETokenizer import BPETokenizer 
from tqdm import tqdm

class Multi30kDatasetEnDe(Dataset):

  def __init__(self,split="train", vocab_en = None, vocab_de = None, max_merges = 5000):

    dataset = load_dataset("bentrevett/multi30k", split=split)
    self.data_en = [x['en'].lower() for x in dataset]
    self.data_de = [x['de'].lower() for x in dataset]

    if vocab_en == None and vocab_de == None:
      self.vocab_en = BPETokenizer()
      self.vocab_en.train(" ".join([x['en'].lower() for x in dataset]), max_merges)

      self.vocab_de = BPETokenizer()
      self.vocab_de.train(" ".join([x['de'].lower() for x in dataset]), max_merges)
      
    else:
      self.vocab_en = vocab_en
      self.vocab_de = vocab_de

    self.data_en = [self.vocab_en.tokenize(x['en'].lower()) for x in tqdm(dataset, desc="Tokenizing English")]
    self.data_de = [self.vocab_de.tokenize(x['de'].lower()) for x in tqdm(dataset, desc="Tokenizing German")]
    
    

  def __len__(self):
    return len(self.data_en)

  def __getitem__(self, idx):
    numeralized_en = [self.vocab_en.sos_id] + self.data_en[idx] + [self.vocab_en.eos_id]
    numeralized_de = [self.vocab_de.sos_id] + self.data_de[idx] + [self.vocab_de.eos_id]
    return torch.tensor(numeralized_de),torch.tensor(numeralized_en)
  
  
  def get_pad_collate(self):

    def pad_collate(batch):
      xx = [ele[0] for ele in batch]
      yy = [ele[1] for ele in batch]
      x_lens = torch.LongTensor([len(x)-1 for x in xx])
      y_lens = torch.LongTensor([len(y)-1 for y in yy])

      xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.vocab_de.pad_id)
      yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.vocab_en.pad_id)

      return xx_pad, yy_pad, x_lens, y_lens
    
    return pad_collate

def getMulti30kDataloadersAndVocabs(config):
  batch_size = config["bs"]
  multi_train = Multi30kDatasetEnDe(split="train", max_merges=config["max_merges"])
  multi_val = Multi30kDatasetEnDe(split="validation", vocab_en=multi_train.vocab_en, vocab_de=multi_train.vocab_de)
  multi_test = Multi30kDatasetEnDe(split="test",  vocab_en=multi_train.vocab_en, vocab_de=multi_train.vocab_de)

  collate = multi_train.get_pad_collate()
  train_loader = DataLoader(multi_train, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=collate, drop_last=True, pin_memory=True, prefetch_factor=4)
  val_loader = DataLoader(multi_val, batch_size=batch_size, num_workers=8,  shuffle=False, collate_fn=collate)
  test_loader = DataLoader(multi_test, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate)

  return train_loader, val_loader, test_loader, {"en":multi_train.vocab_en, "de":multi_train.vocab_de}

if __name__ == "__main__":
  getMulti30kDataloadersAndVocabs()