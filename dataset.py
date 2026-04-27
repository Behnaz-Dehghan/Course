import random

from torch.utils.data import Dataset
import torch
import torchaudio
import os
import pandas as pd
from sklearn.model_selection import train_test_split



# loading tokens from disk/cpu and creating dataset
class FocalTokenDataset(Dataset):
    def __init__(self, csv_path, token_root, vocab, max_sec=3.0, fps=12.5):
        self.df = pd.read_csv(csv_path)
        self.token_root = token_root
        self.vocab = vocab
        #  3 * 12.5 = 37.5 -> 38
        self.max_tokens = int(max_sec * fps)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df.iloc[idx]['mixture_ID']
       
        try:
            src_tokens = torch.load(os.path.join(self.token_root, 'mixture', f"{file_id}.pt"))
            s1_tokens = torch.load(os.path.join(self.token_root, 's1', f"{file_id}.pt"))
            s2_tokens = torch.load(os.path.join(self.token_root, 's2', f"{file_id}.pt"))
        except FileNotFoundError:
            print(f" Warning: Missing or corrupt file {file_id}. Skipping...")
            return self.__getitem__(random.randint(0, len(self.df) - 1))
        #print(f"Token Range: Min = {src_tokens.min().item()}, Max = {src_tokens.max().item()}")
        #clipping 
        if self.token_root.endswith("12_5hz") or self.token_root.endswith("full"):
            curr_len = src_tokens.size(0)
            if curr_len > self.max_tokens:
                start = random.randint(0, curr_len - self.max_tokens)
                src_tokens = src_tokens[start : start + self.max_tokens]
                s1_tokens = s1_tokens[start : start + self.max_tokens]
                s2_tokens = s2_tokens[start : start + self.max_tokens]
                #src_tokens = src_tokens[ : self.max_tokens]
                #s1_tokens = s1_tokens[ : self.max_tokens]
                #s2_tokens = s2_tokens[ : self.max_tokens]
                ''' 
            elif curr_len < self.max_tokens:
                pad_len = self.max_tokens - curr_len
                src_tokens = torch.nn.functional.pad(src_tokens, (0, pad_len), value=self.vocab.pad_id)
                s1_tokens = torch.nn.functional.pad(s1_tokens, (0, pad_len), value=self.vocab.pad_id)
                s2_tokens = torch.nn.functional.pad(s2_tokens, (0, pad_len), value=self.vocab.pad_id)
            '''
        bos = torch.tensor([self.vocab.bos_id], dtype=torch.long)
        eos = torch.tensor([self.vocab.eos_id], dtype=torch.long)
        sep = torch.tensor([self.vocab.sep_id], dtype=torch.long)

        # [BOS, Mixture, EOS]
        src = torch.cat([bos, src_tokens, eos]).long()
        
        # [BOS, S1, SEP, S2, EOS]
        tgt = torch.cat([bos, s1_tokens, sep, s2_tokens, eos]).long()
        #print(f"Dataset __getitem__ - src shape: {src.shape}, tgt shape: {tgt.shape}")
        return src, tgt

# this was used for encodec    
class LibriMixTokenDataset(Dataset):
    def __init__(self, df, wav_root, processor):
        self.df = df
        self.wav_root = wav_root
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['mixture_ID'] + ".wav"

        # Load 16kHz files
        mix, _ = torchaudio.load(os.path.join(self.wav_root, 'mix_clean', fname))
        s1, _  = torchaudio.load(os.path.join(self.wav_root, 's1', fname))
        s2, _  = torchaudio.load(os.path.join(self.wav_root, 's2', fname))

        src, tgt = self.processor.build_sequences(s1, s2, mix)

        return src, tgt
'''    
def collate_fn(batch):
    src_list, tgt_list = zip(*batch)

    # Pad all sequences in the batch to the same length using PAD_ID (1027)
    # Resulting shape: [Batch, Max_Time]
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=1027)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=1027)

    return src_padded, tgt_padded   
'''
def collate_fn(batch, vocab):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=vocab.pad_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_list, batch_first=True, padding_value=vocab.pad_id
    )
    return src_padded, tgt_padded  # [Batch, T]