
import os
import torch
import pandas as pd
import torchaudio
from tqdm import tqdm
import focalcodec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_train-100_mix_clean.csv"
#root_dir = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min"
test_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_test_mix_clean.csv"
dev_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_dev_mix_clean.csv"
#output_token_dir = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\train-100\focal_tokens_12_5hz"
output_token_dir = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\train-100\focal_tokens_50hz_2k_causal-full"

#model = focalcodec.FocalCodec.from_pretrained("lucadellalib/focalcodec_12_5hz").to(device)
model = focalcodec.FocalCodec.from_pretrained("lucadellalib/focalcodec_50hz_2k_causal").to(device) # SR 16/24
model.eval()
#print(model.sample_rate_input, model.sample_rate_output)#16/24kHz
def extract_and_save(csv_file=csv_path, output_token_dir=output_token_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(output_token_dir, exist_ok=True)
    
    for sub in ['s1', 's2', 'mixture']:
        os.makedirs(os.path.join(output_token_dir, sub), exist_ok=True)

    print(" Starting Token Extraction...from: ", csv_file)
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            
            file_id = row['mixture_ID']
            
            paths = {
                'mixture': row['mixture_path'],
                's1': row['source_1_path'],
                's2': row['source_2_path']
            }
            
            for key, path in paths.items():
                target_path = os.path.join(output_token_dir, key, f"{file_id}.pt")
                
                if os.path.exists(target_path): continue
                
                wav, sr = torchaudio.load(path)
                if sr != 16000: 
                    print(f"Resampling {key} from {sr} Hz to 16000 Hz...")
                    wav = torchaudio.functional.resample(wav, sr, 16000)

                # added clipping to 2 seconds 
                # target_len = 2 * 16000
                # if wav.size(1) > target_len:
                #     wav = wav[:, :target_len]
                # we do padding in transformer
                # elif wav.size(1) < target_len:
                #     padding = target_len - wav.size(1)
                #     wav = torch.nn.functional.pad(wav, (0, padding))    

                
                toks = model.sig_to_toks(wav.to(device)) # [1, T]
                
                torch.save(toks.squeeze(0).cpu(), target_path)

if __name__ == "__main__":
    extract_and_save(test_csv, os.path.join(output_token_dir, "test")) 
    extract_and_save(dev_csv, os.path.join(output_token_dir, "dev"))
    extract_and_save(csv_path, os.path.join(output_token_dir, "train"))
    train_csv = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\Libri2Mix\wav16k\min\metadata\mixture_train-100_mix_clean.csv"
    #verify_tokens(train_csv, os.path.join(output_token_dir, "train\s1"), vocab_size=8192)
    # decoding test:
    # token_path = r"C:\Users\umroot\Projects\Speech\LibriMix\train-data\train-100\focal_tokens_50hz_2k_causal\train\s1\19-198-0001_27-123349-0024.pt"
    # tokens = torch.load(token_path, map_location=device)
    # with torch.no_grad():
    #     tokens = tokens.unsqueeze(0)  # [1, T]
    #     reconstructed_wav = model.toks_to_sig(tokens)
 
    # torchaudio.save("test-result-50hz.wav", reconstructed_wav.cpu(), 16000)
    

''' 
import torch
import torchaudio

# Load FocalCodec model
codec = torch.hub.load(
    repo_or_dir="lucadellalib/focalcodec",
    model="focalcodec",
    config="lucadellalib/focalcodec_50hz",
    force_reload=True,  # Fetch the latest FocalCodec version from Torch Hub
)
codec.eval().requires_grad_(False)

# Load and preprocess the input audio
audio_file = "audios/librispeech-dev-clean/251-118436-0003.wav"
sig, sample_rate = torchaudio.load(audio_file)
sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate_input)

# Encode audio into tokens
toks = codec.sig_to_toks(sig)  # Shape: (batch, time)
print(toks.shape)
print(toks)

# Convert tokens to their corresponding binary spherical codes
codes = codec.toks_to_codes(toks)  # Shape: (batch, code_time, log2 codebook_size)    #codebook loss??
print(codes.shape)
print(codes)

# Decode tokens back into a waveform
rec_sig = codec.toks_to_sig(toks)       

# Save the reconstructed audio
rec_sig = torchaudio.functional.resample(rec_sig, codec.sample_rate_output, sample_rate)
torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
''' 