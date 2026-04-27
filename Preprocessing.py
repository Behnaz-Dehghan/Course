import torch, torchaudio

# To use Focal codec tokenizer
import focalcodec


class AudioVocab:
    def __init__(self, codebook_size=1024):
        self.codebook_size = codebook_size
        self.sep_id = codebook_size        # 1024
        self.bos_id = codebook_size + 1    # 1025
        self.eos_id = codebook_size + 2    # 1026
        self.pad_id = codebook_size + 3    # 1027
        self.vocab_size = codebook_size + 4

# Encodec Tokenizer    
class AudioProcessor:
    """
    This class handles an end-to-end preprocessing pipeline from raw waveforms to EnCodec tokens.
    Methods:
    1- clip_audio: Clips audio to a fixed duration. (1.5 seconds in this case)
    2- wav_to_tokens: Converts waveform to EnCodec tokens.
    3- build_sequences: Builds source and target sequences for training.

    Attributes:
        model (nn.Module): The pretrained EnCodec model used for tokenization.
        vocab (AudioVocab): The vocabulary object containing special token IDs (BOS, SEP, EOS).
        device (torch.device): The computation device (cuda/cpu).
        duration_sec (int): The target duration in seconds for all audio clips.
        sample_rate (int): The input sample rate (default 16000 for LibriMix).
        resampler (torchaudio.transforms.Resample): Transform to match EnCodec's 24kHz requirement.
    """
    def __init__(self, model_encodec, vocab, device, duration_sec=1):    # , num_codebooks=1 add num of codebooks as a hyperparameter if you want to experiment with that later
        self.model = model_encodec
        self.vocab = vocab
        self.device = device
        self.duration_sec = duration_sec
        self.sample_rate = 16000 # LibriMix standard
        self.resampler = torchaudio.transforms.Resample(16000, 24000).to(device)

    def clip_audio(self, wav):
        """Standardizes audio to exactly self.duration_sec."""
        target_samples = int(self.duration_sec * self.sample_rate)

        if wav.size(1) > target_samples:
            # Cut to target duration
            return wav[:, :target_samples]
        elif wav.size(1) < target_samples:
            # Pad with zeros if shorter than 3 seconds
            padding = target_samples - wav.size(1)
            return torch.nn.functional.pad(wav, (0, padding))
        return wav

    def wav_to_tokens(self, wav):
        # 1. Move to GPU (wav is already 16kHz here)
        wav = wav.to(self.device)
        # 2. Resample to 24kHz for EnCodec
        wav_24 = self.resampler(wav)

        with torch.no_grad():
            # 3. Encode
            # we use [0][0] to get the tensor from the list of frames
            encoded_frames = self.model.encode(wav_24.unsqueeze(0))
            codes = encoded_frames[0][0] # Shape: [1, K, T]
            #print(f"Encoded shape (K, T): {codes.shape} for input length {wav.size(1)} samples")  #Encoded shape (K, T): torch.Size([1, 2, 75]) for input length 16000 samples
            codes = codes.squeeze(0) # Shape: [K, T]
            #mix_tokens = codes.transpose(0, 1).flatten() # putting codebooks in series is not effiecient and is usually done in decode-only
            #Selected codebook shape (T,): torch.Size([150]) for input length 16000 samples
            #print(f"Selected codebook shape (T,): {mix_tokens.shape} for input length {wav.size(1)} samples")  #Selected codebook shape (T,): torch.Size([75]) for input length 16000 samples
            return codes  #codes[0, 0, :] for 1 codebook 

    def build_sequences(self, s1_wav, s2_wav, mix_wav):
        # --- ADD CLIPPING HERE ---
        s1_wav = self.clip_audio(s1_wav)
        s2_wav = self.clip_audio(s2_wav)
        mix_wav = self.clip_audio(mix_wav)
        # -------------------------

        c1 = self.wav_to_tokens(s1_wav)
        c2 = self.wav_to_tokens(s2_wav)
        src = self.wav_to_tokens(mix_wav)
        '''     
        tgt = torch.cat([
            torch.tensor([self.vocab.bos_id], device=self.device),
            c1,
            torch.tensor([self.vocab.sep_id], device=self.device),
            c2,
            torch.tensor([self.vocab.eos_id], device=self.device)
        ])'''
        # For more codebooks:
        assert c1.shape == c2.shape == src.shape, "Tokens for both speakers must have the same shape" #[2,75] for 1 second of audio with 2 codebooks
        K, T = c1.shape
        bos = torch.full((K, 1), self.vocab.bos_id, device=self.device)
        eos = torch.full((K, 1), self.vocab.eos_id, device=self.device)
        sep = torch.full((K, 1), self.vocab.sep_id, device=self.device)
        tgt = torch.cat([bos, c1, sep, c2, eos], dim=1)
        src = torch.cat([bos, src, eos], dim=1)
        return src, tgt

# this is not used!        
class FocalCodecProcessor:
    """
    Key differences from EnCodec:
    - Single codebook only (tokens shape: [T] not [K, T])
    - Input: 16kHz (no resampling needed for LibriMix!)
    - Vocab size: 2^12 = 4096 -> set it in main !
    - 50 tokens/sec at 50hz variant or 12Hz variant ?
    """
    def __init__(self, vocab, device, variant="lucadellalib/focalcodec_12hz", duration_sec=2):
        self.device = device
        self.vocab = vocab
        self.duration_sec = duration_sec
        self.sample_rate = 16000  # FocalCodec 16kHz input — matches LibriMix directly!

        self.model = focalcodec.FocalCodec.from_pretrained(variant)
        self.model.eval().requires_grad_(False).to(device)

    def clip_audio(self, wav):
        target_samples = int(self.duration_sec * self.sample_rate)
        if wav.size(1) > target_samples:
            return wav[:, :target_samples]
        elif wav.size(1) < target_samples:
            padding = target_samples - wav.size(1)
            return torch.nn.functional.pad(wav, (0, padding))
        return wav

    def wav_to_tokens(self, wav):
        """
        wav: [1, T] at 16kHz
        returns: [T_tokens] — 1D, single codebook
        """
        wav = wav.to(self.device)
        sig = torchaudio.functional.resample(wav, self.sample_rate, self.model.sample_rate_input)
        with torch.no_grad():
            toks = self.model.sig_to_toks(sig)  # shape: [1, T_tokens]
            print(f"FocalCodec tokens shape: {toks.shape} for input length {wav.size(1)} samples")
        return toks.squeeze(0)  # [T_tokens]

    def tokens_to_wav(self, tokens):
        """
        tokens: [T_tokens] 1D tensor
        returns: [T_wav] waveform at 16kHz
        """
        toks = tokens.unsqueeze(0).to(self.device)  # [1, T_tokens]
        with torch.no_grad():
            rec_sig = self.model.toks_to_sig(toks)  # [1, T_wav] at sample_rate_output
        # Resample back to 16kHz if needed
        wav = torchaudio.functional.resample(
            rec_sig.squeeze(0).unsqueeze(0), 
            self.model.sample_rate_output, 
            self.sample_rate
        )
        return wav.squeeze()

    def build_sequences(self, s1_wav, s2_wav, mix_wav):
        s1_wav = self.clip_audio(s1_wav)
        s2_wav = self.clip_audio(s2_wav)
        mix_wav = self.clip_audio(mix_wav)

        c1  = self.wav_to_tokens(s1_wav)   # [T]
        c2  = self.wav_to_tokens(s2_wav)   # [T]
        src = self.wav_to_tokens(mix_wav)  # [T]

        bos = torch.tensor([self.vocab.bos_id], device=self.device)
        eos = torch.tensor([self.vocab.eos_id], device=self.device)
        sep = torch.tensor([self.vocab.sep_id], device=self.device)

        tgt = torch.cat([bos, c1, sep, c2, eos],dim=0)  # [2T + 3]
        src = torch.cat([bos, src, eos],dim=0)           # [T + 2]
        return src, tgt    
