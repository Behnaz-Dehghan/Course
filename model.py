import torch
import torch.nn as nn
import math
#from speechbrain.lobes.augment import SpecAugment

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        # We add dropout here to ensure the model doesn't become too dependent on specific absolute positions, making it more robust to different audio lengths.
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #print(f"Positional Encoding shape: {pe.shape}")  # Positional Encoding shape: torch.Size([2048, 256])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq, d_model]
        #print(f"Input to PositionalEncoding shape: {x.shape}")  # Input to PositionalEncoding shape: torch.Size([32, 150, 256])
        x = x + self.pe[:, :x.size(1), :]
        return x #self.dropout(x)
    
class SimpleSpecAugment(nn.Module):
    def __init__(self, dim_mask_width=10, time_mask_width=5):
        super().__init__()
        self.feat_mask_width = dim_mask_width   # 256
        self.time_mask_width = time_mask_width  #100

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        if not self.training:
            return x
        
        batch, time, d_model = x.shape
        
        # feature masking
        f = torch.randint(0, self.feat_mask_width, (1,)).item()
        f0 = torch.randint(0, d_model - f, (1,)).item()
        x[:, :, f0:f0+f] = 0
        
        # Time masking
        t = torch.randint(0, self.time_mask_width, (1,)).item()
        t0 = torch.randint(0, time - t, (1,)).item()
        x[:, t0:t0+t, :] = 0
        
        return x
    
class TransformerSeq2Seq(nn.Module):
    """
    Encoder-Decoder Transformer with autoregressive inference for audio separation.
    src: Mixture tokens [Batch, T_src, d_model]
    tgt: Speaker tokens [Batch, T_tgt, d_model] 
    """
    def __init__(self, vocab, d_model=256, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab.vocab_size
        self.pad_idx = vocab.pad_id
        
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=2048)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
            ,norm_first=True #added layer norm before attention and feedforward, can help with training stability and convergence
        )
        self.fc = nn.Linear(d_model, self.vocab_size)

        self.spec_aug = SimpleSpecAugment() 

    def forward(self, src, tgt):
        # 1. Embeddings + Position
        #print(f"Before embedding: {src.shape}")
        src_emb = self.pos_enc(self.embedding(src))
        tgt_emb = self.pos_enc(self.embedding(tgt))

        # 2. Generate Causal Mask (Look-ahead mask)
        # Prevents position 'i' from seeing position 'i+1'
        T = tgt.size(1)
        tgt_mask = torch.triu(  #Triangular Upper, 1 (True) means "Mask this position out"
            torch.ones(T, T, dtype=torch.bool),
            diagonal=1
        ).to(tgt.device)
        #tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        # 3. Padding Masks (True means the value will be IGNORED)
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (tgt == self.pad_idx)

        #add augmentation on token embeddings
        if self.training:
            src_emb = self.spec_aug(src_emb)

        # 4. Transformer Forward
        # memory_key_padding_mask tells the decoder to ignore PAD in the mixture
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask 
        )

        return self.fc(out)
    
# this is not used
# in case of Encodec with 2 codebooks 
class TransformerSeq2Seq_multi(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab.vocab_size
        self.pad_idx = vocab.pad_id
        self.n_codebooks = 2  # EnCodec set param later
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, d_model) for _ in range(self.n_codebooks)
        ]) # Separate embedding layers for each codebook, allowing the model to learn distinct representations for each.
        self.emb_norm = nn.LayerNorm(d_model)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=2048)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, self.vocab_size)

    def forward(self, src, tgt):

        # Summing codebook embeddings is a common practice to keep sequence length manageable, but it can lead to information loss.
        # src/tgt shape: [Batch, K, T]
        #src_emb = torch.zeros(batch_size, T, self.d_model, device=src.device)
        src_emb = self.embeddings[0](src[:, 0, :])
        tgt_emb = self.embeddings[0](tgt[:, 0, :])
        for k in range(1,self.n_codebooks):
            src_emb += self.embeddings[k](src[:, k, :])
            tgt_emb += self.embeddings[k](tgt[:, k, :])
        src_emb = self.emb_norm(src_emb) # layer norm or averaging is recommended
        tgt_emb = self.emb_norm(tgt_emb)
        # 1. Embeddings + Position
        #src_emb = self.pos_enc(self.embedding(src))
        #tgt_emb = self.pos_enc(self.embedding(tgt))
        src_emb = self.pos_enc(src_emb)
        tgt_emb = self.pos_enc(tgt_emb)
        # 2. Generate Causal Mask (Look-ahead mask)
        # Prevents position 'i' from seeing position 'i+1'
        #T = tgt.size(1)
        #tgt_mask = torch.triu(  #Triangular Upper, 1 (True) means "Mask this position out"
        #    torch.ones(T, T, dtype=torch.bool),
        #    diagonal=1
        #).to(tgt.device)
        T = tgt_emb.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)
        
        # 3. Padding Masks (True means the value will be IGNORED)
        #src_pad_mask = (src == self.pad_idx)
        #tgt_pad_mask = (tgt == self.pad_idx)
        src_pad_mask = (src[:, 0, :] == self.pad_idx) # Shape: [Batch, T]
        tgt_pad_mask = (tgt[:, 0, :] == self.pad_idx) # Shape: [Batch, T]

        # 4. Transformer Forward
        # memory_key_padding_mask tells the decoder to ignore PAD in the mixture
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask 
        )

        return self.fc(out)
    








# class TransformerSeq2Seq_improved(nn.Module):

#     def __init__(self, vocab, d_model=256, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1):
#         super().__init__()
#         self.vocab_size = vocab.vocab_size
#         self.pad_idx = vocab.pad_id
        
#         self.embedding = nn.Embedding(self.vocab_size, d_model)
#         # instead of nn.embedding, we could use pretrained focalcodecwights
#         codec_embeddings = processor.model.quantizer.embed  
#         self.embedding.weight.data = codec_embeddings
#         self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=2048)
        
#         self.transformer = nn.Transformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=dim_ff,
#             dropout=dropout,
#             batch_first=True
#             ,norm_first=True #added layer norm before attention and feedforward, can help with training stability and convergence
#         )
#         self.fc = nn.Linear(d_model, self.vocab_size)

#     def forward(self, src, tgt):
#         # 1. Embeddings + Position
#         #print(f"Before embedding: {src.shape}")
#         src_emb = self.pos_enc(self.embedding(src))
#         tgt_emb = self.pos_enc(self.embedding(tgt))

#         # 2. Generate Causal Mask (Look-ahead mask)
#         # Prevents position 'i' from seeing position 'i+1'
#         T = tgt.size(1)
#         tgt_mask = torch.triu(  #Triangular Upper, 1 (True) means "Mask this position out"
#             torch.ones(T, T, dtype=torch.bool),
#             diagonal=1
#         ).to(tgt.device)
#         #tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
#         # 3. Padding Masks (True means the value will be IGNORED)
#         src_pad_mask = (src == self.pad_idx)
#         tgt_pad_mask = (tgt == self.pad_idx)

#         # 4. Transformer Forward
#         # memory_key_padding_mask tells the decoder to ignore PAD in the mixture
#         out = self.transformer(
#             src_emb,
#             tgt_emb,
#             tgt_mask=tgt_mask,
#             src_key_padding_mask=src_pad_mask,
#             tgt_key_padding_mask=tgt_pad_mask,
#             memory_key_padding_mask=src_pad_mask 
#         )

#         return self.fc(out)
