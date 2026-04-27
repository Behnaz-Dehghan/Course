import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import soundfile as sf
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import IPython.display as ipd

from Preprocessing import AudioProcessor, AudioVocab, FocalCodecProcessor
from dataset import FocalTokenDataset, LibriMixTokenDataset, collate_fn
from model import TransformerSeq2Seq
from train import NoamOpt, beam_search, calculate_sisnr, debug_overfitting, generate_tokens, load_model, plot_training_history, test_model, test_teacher_forcing, tokens_to_audio, train_model
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from functools import partial
import focalcodec 
#from Conformer import ConformerModel, ParallelConformerSeparation
random.seed(0)
torch.manual_seed(0)

#-----argpars
parser = argparse.ArgumentParser()
#parser.add_argument('--duration', type=float, default=1.5, help='Duration in seconds')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
#parser.add_argument('--n_codebooks', type=int, default=2, help='Number of codebooks to use')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8) 
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dim_ff', type=int, default=1024) 
parser.add_argument('--tokenizer', type=str, default="focal")#"encodec")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.d_model = 256#128#512#256
args.nhead = 4#8
args.num_layers = 2
args.dim_ff = 1024 #256 #2048#1024
args.dropout = 0.3 # first tried 0.1
if args.tokenizer == "encodec":
    '''
    # Using ENCODEC for tokenization
    model_encodec = EncodecModel.encodec_model_24khz()
    model_encodec.set_target_bandwidth(1.5) # 1.5 kbps gives 2 codebooks      6kbps gives 8 codebooks
    model_encodec.to(device)
    vocab = AudioVocab()
    processor = AudioProcessor(model_encodec, vocab, device)
    '''
else:
    #Focalcodec
    print("Using FocalCodec tokenizer")
    vocab = AudioVocab(codebook_size=2048) #8192 for 12.5hz, 2048 for 50hz 
    #processor = FocalCodecProcessor(vocab, device, variant="lucadellalib/focalcodec_12hz", duration_sec=1)
    #model = TransformerSeq2Seq(vocab, d_model=256, nhead=8, num_layers=2, dim_ff=1024, dropout=0.1).to(device)
    #model = TransformerSeq2Seq(vocab, d_model=512, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1).to(device)
    model = TransformerSeq2Seq(vocab, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_ff=args.dim_ff, dropout=args.dropout).to(device)
    #model  = ConformerModel(vocab.vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
    #model = ParallelConformerSeparation(vocab_size=1024, d_model=256, nhead=8, num_layers=4).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")
print(argparse.Namespace(**vars(args)))

train_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_train-100_mix_clean"+"_ordered.csv"
test_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_test_mix_clean"+"_ordered.csv"
dev_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_dev_mix_clean"+"_ordered.csv"
#token_dir = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\train-100\focal_tokens_12_5hz"
token_dir = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\train-100\focal_tokens_50hz_2k_causal"#-full"

num_train = 10000
num_val = 1000
num_test = 100

train_ds = FocalTokenDataset(train_csv, os.path.join(token_dir, "train"), vocab, max_sec=3.0, fps=50)
dev_ds = FocalTokenDataset(dev_csv, os.path.join(token_dir, "dev"), vocab, max_sec=3.0, fps=50)
test_ds = FocalTokenDataset(test_csv, os.path.join(token_dir, "test"), vocab, max_sec=3.0, fps=50)

#train_ds = Subset(train_ds, np.random.choice(len(train_ds), num_train, replace=False))
#dev_ds = Subset(dev_ds, np.random.choice(len(dev_ds), num_val, replace=False))
#test_ds = Subset(test_ds, np.random.choice(len(test_ds), num_test, replace=False))
my_collate = partial(collate_fn, vocab=vocab)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
val_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)
print(f"Train samples: {len(train_ds)}, Val samples: {len(dev_ds)}, Test samples: {len(test_ds)}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")  
print(f"Example batch shapes - src: {next(iter(train_loader))[0].shape}, tgt: {next(iter(train_loader))[1].shape}")
print(f"Example batch - src tokens: {next(iter(train_loader))[0][0][:10]}, tgt tokens: {next(iter(train_loader))[1][0][:10]}")
print(f"Vocabulary - BOS: {vocab.bos_id}, SEP: {vocab.sep_id}, EOS: {vocab.eos_id}, PAD: {vocab.pad_id}")   
#optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=1e-5)
#first tried with this:
#optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)#1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='min', 
#     factor=0.5,   
#     patience=3,#2, 
#     threshold=0.01,#added since loss changes ar small 
#     verbose=True
# )
# seocnd time, tried with noamopt:
#base_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
base_optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0, 
    betas=(0.9, 0.98), 
    eps=1e-9, 
    weight_decay=0.01 # adds reguralization to prevent overfitting,(since there was overfitting in early epochs stopping on loss 5, acc 15%) since dataset is small
)
optimizer = NoamOpt(
    model_size=args.d_model, 
    factor=1, 
    warmup=2000,#4000, #2000 for smaller dmodel
    optimizer=base_optimizer
)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.1) # epsilon = 0.1
print(device)
#debug_overfitting(model, train_loader, optimizer, criterion, device)
 


# -------------------------uncomment for trinaing-----------------------
# history = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader, 
#     optimizer=optimizer,
#     scheduler=None,
#     criterion=criterion,
#     device=device,
#     vocab=vocab,
#     epochs=200,
#     args=args,
#     save_dir="checkpoint-focal-50hz-reordered-specaug", 
#     start_epoch=0,
#     history=None
# )
# plot_training_history(history, save_path="checkpoint-focal-50hz-reordered-specaug/training.png")
#plot_training_history(history, save_path="checkpoint-focal-50hz-PIT/training.png") #256 4 head, 2 layers, dim_ff 1024, dropout 0.3, noamopt, PIT loss with sep token loss, 50hz tokens,2s
# plot_training_history(history, save_path="checkpoint-focal-50hz-shrink-3s/training.png")

#-----------lodaing best model for inference--------------
#checkpoint_path = "checkpoint-focal-50hz-1/transformer_epoch_50.pth" 
#checkpoint_path = "checkpoint-focal-50hz-shrink/best_model.pth"
checkpoint_path = "checkpoint-focal-50hz-specaug/best_model.pth"
model, history = load_model(checkpoint_path, model, device=device)
#checkpoint = torch.load(checkpoint_path, map_location='cpu')
#print(checkpoint['config'])
 
model.to(device)
model.eval()
batch = next(iter(val_loader))
src = batch[0].to(device)
focal_model = focalcodec.FocalCodec.from_pretrained("lucadellalib/focalcodec_50hz_2k_causal").to(device) # SR 16/24
focal_model.eval()
test_model(model, test_loader, vocab,focal_model, criterion, device, save_dir="checkpoint-focal-50hz-specaug", beamsearch=True)
src_samp, original_samp, recon_samp = test_teacher_forcing(model, batch, vocab, device)

wav_recon_s1, wav_recon_s2 = tokens_to_audio(recon_samp.unsqueeze(0), vocab, focal_model)
wav_recon_s1 = torchaudio.functional.resample(wav_recon_s1, 24000, 16000)
wav_recon_s2 = torchaudio.functional.resample(wav_recon_s2, 24000, 16000)
wav_orig_s1, wav_orig_s2 = tokens_to_audio(original_samp.unsqueeze(0), vocab, focal_model)
wav_orig_s1 = torchaudio.functional.resample(wav_orig_s1, 24000, 16000)
wav_orig_s2 = torchaudio.functional.resample(wav_orig_s2, 24000, 16000)
if wav_recon_s1 is not None:
    torchaudio.save("output_s1.wav", wav_recon_s1.unsqueeze(0), 16000)
    print(" Speaker 1 saved as output_s1.wav")
if wav_recon_s2 is not None:
    torchaudio.save("output_s2.wav", wav_recon_s2.unsqueeze(0), 16000)
    print(" Speaker 2 saved as output_s2.wav")
if wav_orig_s1 is not None:
    torchaudio.save("original_s1.wav", wav_orig_s1.unsqueeze(0), 16000)
    print(" Original Speaker 1 saved as original_s1.wav")     
if wav_orig_s2 is not None:
    torchaudio.save("original_s2.wav", wav_orig_s2.unsqueeze(0), 16000)
    print(" Original Speaker 2 saved as original_s2.wav")

# greedy inference
print("Generating tokens... .")
generated_tokens = generate_tokens(model, src[0:1], vocab, max_len=205) #first sample
wav_s1, wav_s2 = tokens_to_audio(generated_tokens, vocab, focal_model, device=device)
wav_s1 = torchaudio.functional.resample(wav_s1, 24000, 16000)
wav_s2 = torchaudio.functional.resample(wav_s2, 24000, 16000)
if wav_s1 is not None:
    torchaudio.save("generated_s1.wav", wav_s1.unsqueeze(0), 16000)
    print(" Speaker 1 saved as generated_s1.wav")
if wav_s2 is not None:
    torchaudio.save("generated_s2.wav", wav_s2.unsqueeze(0), 16000)
    print(" Speaker 2 saved as generated_s2.wav")

#----beamsearch inference----
generated_tokens = beam_search(model, src[0:1], vocab, max_len=205) #first sample
wav_s1, wav_s2 = tokens_to_audio(generated_tokens, vocab, focal_model, device=device)
wav_s1 = torchaudio.functional.resample(wav_s1, 24000, 16000)
wav_s2 = torchaudio.functional.resample(wav_s2, 24000, 16000)
if wav_s1 is not None:
    torchaudio.save("beam_generated_s1.wav", wav_s1.unsqueeze(0), 16000)
    print(" Speaker 1 saved as beam_generated_s1.wav")
if wav_s2 is not None:
    torchaudio.save("beam_generated_s2.wav", wav_s2.unsqueeze(0), 16000)
    print(" Speaker 2 saved as beam_generated_s2.wav")












""" history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader, 
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    vocab=vocab,
    epochs=50,
    args=args,
    save_dir="checkpoint-focal-50hz-1"
) """
#checkpoint_path = 'checkpoint-focal-2/transformer_epoch_50.pth'#'checkpoint-focal-50hz-1/transformer_epoch_10.pth'
#####checkpoint_path = 'checkpoint-focal-50hz-3/best_model.pth'
# checkpoint = torch.load(checkpoint_path, map_location=device)
# history = checkpoint['history']
# plot_training_history(history, save_path="training_history.png")
 
#checkpoint_path = 'checkpoint-focal-1/transformer_epoch_15.pth'
""" if os.path.exists(checkpoint_path):
    print(f" Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    history = checkpoint['history']
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Current LR: {optimizer.param_groups[0]['lr']}")
    print(f" Resuming from Epoch {start_epoch}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-4
    model.train()  
    
else:
    print(" No checkpoint found. Starting from scratch.")
    start_epoch = 0 """

# history = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader, 
#     optimizer=optimizer,
#     scheduler=scheduler,
#     criterion=criterion,
#     device=device,
#     vocab=vocab,
#     epochs=200,
#     args=args,
#     save_dir="checkpoint-focal-50hz-3",
#     start_epoch=start_epoch,
#     history=history
# ) 
# 