import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):

    # kq_dim  is the  dimension  of keys  and  values. Linear  layers  should  be usedto  project  inputs  to these  dimensions.
    def __init__(self, q_input_dim, cand_input_dim, v_dim, kq_dim=64):
        super().__init__()
        ##################################
        #  Q17
        ##################################

        #TODO

    #hidden  is h_t^{d} from Eq. (11)  and has  dim => [batch_size , dec_hid_dim]
    #encoder_outputs  is the  word  representations  from Eq. (6)
    # and has dim => [batch_size, src_len , enc_hid_dim * 2]
    def forward(self, hidden, encoder_outputs):
        ##################################
        #  Q17
        ##################################

        #TODO

        return attended_val, alpha



class Dummy(nn.Module):

    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim
        
    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], self.v_dim) ).to(hidden.device)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device)
        return zout, zatt

class MeanPool(nn.Module):

    def __init__(self, cand_input_dim, v_dim):
        super().__init__()
        self.linear = nn.Linear(cand_input_dim, v_dim)

    def forward(self, hidden, encoder_outputs):

        encoder_outputs = self.linear(encoder_outputs)
        output = torch.mean(encoder_outputs, dim=1)
        alpha = F.softmax(torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device), dim=-1)

        return output, alpha

class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab_len, emb_dim, enc_hid_dim, dropout=0.5):
        super().__init__()

        ##################################
        #  Q15
        ##################################
        self.embed = nn.Embedding(src_vocab_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.GRU = nn.GRU(input_size=emb_dim,hidden_size=enc_hid_dim,num_layers=1,bidirectional=True,batch_first=True)
        self.enc_hid_dim = enc_hid_dim
        

    def forward(self, src, src_lens):
        ##################################
        #  Q15
        ##################################
        em = self.dropout(self.embed(src))
        word_representations, _ = self.GRU(em)
        lastforward = word_representations[torch.arange(src.shape[0],device=src.device),src_lens - 1,:self.enc_hid_dim]
        firstbackward = word_representations[:, 0, self.enc_hid_dim:]
        sentence_cat = torch.cat([lastforward,firstbackward], dim=-1)
        return word_representations, sentence_cat


class Decoder(nn.Module):
    def __init__(self, trg_vocab_len, emb_dim, dec_hid_dim, attention, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(trg_vocab_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.GRU = nn.GRU(input_size=emb_dim,hidden_size=dec_hid_dim,num_layers=1,batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(dec_hid_dim, dec_hid_dim), nn.GELU(), nn.Linear(dec_hid_dim, trg_vocab_len),)

        ##################################
        #  Q16
        ##################################

    def forward(self, input, hidden, encoder_outputs):
        ##################################
        #  Q16
        ##################################
        em = self.dropout(self.embed(input))
        em = em.unsqueeze(1)
        hidden = hidden.unsqueeze(0)
        _, h = self.gru(em, hidden)
        h = h.squeeze(0)
        attended_feature, alphas = self.attention(h, encoder_outputs)
        new_hidden = h + attended_feature
        out = self.classifier(new_hidden)
        #Output prediction (scores for each word), the updated hidden state, and the attention map (for visualization)
        return new_hidden, out, alphas

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, kq_dim, attention, dropout=0.5):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = BidirectionalEncoder(src_vocab_size, embed_dim, enc_hidden_dim, dropout=dropout)
        self.enc2dec = nn.Sequential(nn.Linear(enc_hidden_dim*2, dec_hidden_dim), nn.GELU())

        if attention == "none":
            attn_model = Dummy(dec_hidden_dim)
        elif attention == "mean":
            attn_model = MeanPool(2*enc_hidden_dim, dec_hidden_dim)
        elif attention == "dotproduct":
            attn_model = DotProductAttention(dec_hidden_dim, 2*enc_hidden_dim, dec_hidden_dim, kq_dim)

        
        self.decoder = Decoder(trg_vocab_size, embed_dim, dec_hidden_dim, attn_model, dropout=dropout)
        



    def translate(self, src, src_lens, sos_id=1, max_len=50):
        
        #tensor to store decoder outputs and attention matrices
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)
        attns = torch.zeros(src.shape[0], max_len, src.shape[1]).to(src.device)


        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device)*sos_id

        ##################################
        #  Q19
        ##################################

        #TODO

        return outputs, attns

    def forward(self, src, trg, src_lens):

        #tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg_len, self.trg_vocab_size).to(src.device)

        ##################################
        #  Q18
        ##################################

        #TODO

        return outputs