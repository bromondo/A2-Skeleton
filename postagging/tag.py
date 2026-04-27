import pickle
import torch
from models.PoSGRU import PoSGRU
with open("./chkpts/S5zIj3_UDPOS_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
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

sentence = input("Enter a sentence Please:")
words = sentence.lower().split()
Numberedwords = vocab.numeralizeSentence(words)
Tensornumberedwords = torch.LongTensor(Numberedwords).unsqueeze(0)

model = PoSGRU(vocab_size=vocab.lenWords(), 
                 embed_dim=config["embed_dim"], 
                 hidden_dim=config["hidden_dim"], 
                 num_layers=config["layers"],
                 output_dim= vocab.lenLabels(),
                 residual=config["residual"],
                 embed_init=None)

model.load_state_dict(torch.load("./chkpts/S5zIj3_UDPOS_model.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    out = model(Tensornumberedwords)
