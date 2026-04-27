import os
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import numpy as np
#from speechbrain.nnet.losses import PitWrapper

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, vocab, epochs=50, args=None, save_dir="checkpoints",start_epoch=0, history=None):
    """
    Executes the training and validation loops.

    This function implements Teacher Forcing, Gradient Clipping, and checkpoint saving. 

    Args:
        model (nn.Module): The Transformer architecture to be trained.
        train_loader (DataLoader): DataLoader containing training (Mixture, Target) pairs.
        val_loader (DataLoader): DataLoader for validation set.
        optimizer (torch.optim.Optimizer): Adam
        criterion (nn.Module): CrossEntropyLoss with ignore_index.
        device (torch.device): cuda
        vocab (AudioVocab): The vocabulary containing audio token ideces andspecial tokens (pad_id, bos_id).
        epochs (int): Total number of full passes over the training dataset.
        save_dir (str): Directory path where .pth checkpoints will be stored.

    Returns:
        dict: A dictionary containing 'train_loss', 'val_loss', and 'val_acc' lists 
              for each epoch, used for plotting.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if history is None:
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    print(f" Training started on {device}...")
   
    early_stopping = EarlyStopping(patience=30, path=os.path.join(save_dir, 'best_model.pth'))
    for epoch in range(start_epoch, epochs):
        # --- TRAINING ---
        model.train()
        total_train_loss = 0
        print(f"Number of batches in train_loader: {len(train_loader)}")
        print(f"Number of batches in val_loader: {len(val_loader)}")
        for batch in train_loader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            # Teacher Forcing Shift
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            # in case of multiple codebooks
            #tgt_input = tgt[:, :, :-1] 
            #tgt_labels = tgt[:, :, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input) # [Batch, Seq, Vocab]
            
            # Reshape for CrossEntropy: [Batch * Seq, Vocab]
            # loss = criterion(output.reshape(-1, output.size(-1)), tgt_labels.reshape(-1))

            # PIT loss
            sep_idx = (tgt_labels[0] == vocab.sep_id).nonzero(as_tuple=True)[0].item()
            tgt_s1 = tgt_labels[:, :sep_idx]
            tgt_s2 = tgt_labels[:, sep_idx+1:-1] #eos
            #sep_idx = (output[0] == vocab.sep_id).nonzero(as_tuple=True)[0].item()
            output_s1 = output[:, :sep_idx, :] 
            output_s2 = output[:, sep_idx+1:-1, :]
            #min_len = min(output_s1.size(1), output_s2.size(1), tgt_s1.size(1), tgt_s2.size(1))
            loss1 = criterion(output_s1.reshape(-1, output.size(-1)), tgt_s1.reshape(-1)) + \
             criterion(output_s2.reshape(-1, output.size(-1)), tgt_s2.reshape(-1))
            loss2 = criterion(output_s1.reshape(-1, output.size(-1)), tgt_s2.reshape(-1)) + \
             criterion(output_s2.reshape(-1, output.size(-1)), tgt_s1.reshape(-1))
            loss = torch.min(loss1, loss2)  # PIT: choose the permutation with lower loss
            sep_loss = criterion(output[:, sep_idx, :], tgt_labels[:, sep_idx])
            eos_loss = criterion(output[:, -1, :], tgt_labels[:, -1])
            loss += (sep_loss + eos_loss)* 0.5  # give less weight to SEP token 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # try 
            optimizer.step()
            
            total_train_loss += loss.item()

        # --- VALIDATION & ACCURACY ---
        model.eval()
        total_val_loss = 0
        epoch_accuracy = 0
        total_correct_tokens = 0
        total_real_tokens = 0
        total_sep_correct = 0
        total_sep_expected = 0
        val_si_sdrs = []
        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                
                tgt_input, tgt_labels = tgt[:, :-1], tgt[:, 1:]
                output = model(src, tgt_input) 
                
                # Loss
                #v_loss = criterion(output.reshape(-1, output.size(-1)), tgt_labels.reshape(-1))
                # PIT loss
                sep_idx = (tgt_labels[0] == vocab.sep_id).nonzero(as_tuple=True)[0].item()
                tgt_s1 = tgt_labels[:, :sep_idx]
                tgt_s2 = tgt_labels[:, sep_idx+1:-1] #eos
                #sep_idx = (output[0] == vocab.sep_id).nonzero(as_tuple=True)[0].item()
                output_s1 = output[:, :sep_idx, :] 
                output_s2 = output[:, sep_idx+1:-1, :]
                #min_len = min(output_s1.size(1), output_s2.size(1), tgt_s1.size(1), tgt_s2.size(1))
                loss1 = criterion(output_s1.reshape(-1, output.size(-1)), tgt_s1.reshape(-1)) + \
                criterion(output_s2.reshape(-1, output.size(-1)), tgt_s2.reshape(-1))
                loss2 = criterion(output_s1.reshape(-1, output.size(-1)), tgt_s2.reshape(-1)) + \
                criterion(output_s2.reshape(-1, output.size(-1)), tgt_s1.reshape(-1))
                v_loss = torch.min(loss1, loss2)  # PIT: choose the permutation with lower loss
                sep_loss = criterion(output[:, sep_idx, :], tgt_labels[:, sep_idx])
                eos_loss = criterion(output[:, -1, :], tgt_labels[:, -1])
                v_loss += (sep_loss + eos_loss)* 0.5  # give less weight to SEP token
                total_val_loss += v_loss.item()
                
                # Token Accuracy (Ignoring Padding)
                predicted_indices = torch.argmax(output, dim=-1)
                mask = (tgt_labels != vocab.pad_id)
                correct = (predicted_indices == tgt_labels) & mask
                # Sum correct tokens / Sum real tokens
                #epoch_accuracy += (correct.sum().item() / mask.sum().item())
                total_correct_tokens += correct.sum().item()
                total_real_tokens += mask.sum().item()
                # see if SEP token is predicted 
                true_sep_mask = (tgt_labels == vocab.sep_id) # ground- truth Sep positions
                num_true_sep = true_sep_mask.sum().item()
                #sep_mask = (predicted_indices == vocab.sep_id).any(dim=1)
                #if sep_mask.sum() == 0:
                #    print("SEP token NOT predicted in validation batch!") 
                correct_sep = (predicted_indices[true_sep_mask] == vocab.sep_id).sum().item() 
                #print(correct_sep, num_true_sep)
                total_sep_correct += correct_sep   
                total_sep_expected += num_true_sep 
                #total_sep = sep_mask.sum().item()
                ''' 
                if num_true_sep > 0:
                    if correct_sep == 0:
                        print("SEP token NEVER predicted correctly in this batch!")
                    else:
                        pass
                        #sep_acc = (correct_sep / total_sep) * 100 
                        #print(f"SEP Accuracy: {sep_acc:.2f}% ({correct_sep}/{total_sep})")
                '''
                #print(f"--- Structure Check: {total_sep}/{src.size(0)} samples generated at least one SEP ---")

        # Metrics calculation
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = (total_correct_tokens / total_real_tokens) * 100 
        avg_sep_acc = total_sep_correct / total_sep_expected * 100
        
        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        early_stopping(avg_val_loss, model, optimizer, epoch, history, scheduler)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

        print(f"Epoch [{epoch+1}/{epochs}] | Loss (T/V): {avg_train_loss:.4f}/{avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        print(f"--- Structure Check: SEP Token Accuracy: {avg_sep_acc:.2f}% ---")
        # --- SAVE CHECKPOINT EVERY 5 EPOCHS ---
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"transformer_epoch_{epoch+1}.pth")

            config = {
                'num_layers': args.num_layers,
                'd_model': args.d_model,
                'nhead': args.nhead,
                'dim_feedforward': args.dim_ff,
                'lr': optimizer.param_groups[0]['lr'],
                'batch_size': args.batch_size
            }

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
                ,'config': config,
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
            }, checkpoint_path)
            print(f" Checkpoint saved: {checkpoint_path}")

        if scheduler is not None:
            scheduler.step(avg_val_loss)
    return history


def load_model(checkpoint_path, model, optimizer=None,device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # If optimizer is provided, we are RESUMING training
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f" Resuming from Epoch {checkpoint['epoch']}. Ready to continue training!")
        return model, optimizer, checkpoint['history']
    
    # If no optimizer, we are just TESTING
    model.eval()
    print(f" Model loaded from Epoch {checkpoint['epoch']} for testing.")
    return model, checkpoint['history']

def test_teacher_forcing(model, batch, vocab, device):
    model.eval()
    #src, tgt = next(iter(val_loader)) #first batch
    src, tgt = batch
    src = src.to(device)
    tgt = tgt.to(device)
    print(tgt)
    with torch.no_grad():
        tgt_input = tgt[:, :-1] 
        output = model(src, tgt_input)
        predicted_tokens = torch.argmax(output, dim=-1)
        #reconstructed_tgt = torch.cat([tgt[:, 0:1], predicted_tokens], dim=1)  # remove this!!
        print("Predicted Tokens:", predicted_tokens)
    return src[0], tgt[0], predicted_tokens[0]

# inference (greedy)
def generate_tokens(model, src, vocab, max_len=800):
    """This function runs inference on a batch and generates a target tokens in the form of <bos>...<sep>...<eos> given a source input."""
    model.eval()
    device = src.device
    batch_size = src.size(0)

    # Start with the [BOS] token for the decoder
    # Shape: [Batch, 1]
    tgt = torch.tensor([[vocab.bos_id]], device=device).repeat(batch_size, 1)

    with torch.no_grad():
        '''
        for i in range(max_len):
            # Generate the prediction for the current sequence
            output = model(src, tgt)

            # Take the last token's logits: [Batch, Vocab]
            next_token_logits = output[:, -1, :]

            # Greedy choice: Take the index with the highest probability
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            # Append to the sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if ALL sequences in the batch have produced <eos>
            if (next_token == vocab.eos_idx).all():
                break
                '''

        # Initialize a mask of 'False' for each item in the batch
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_len):
                output = model(src, tgt)
                next_token = torch.argmax(output[:, -1, :], dim=-1) #greedy

                # If a sequence is already finished, replace its prediction with <pad>
                next_token = torch.where(finished, vocab.pad_id, next_token)

                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

                # Update the finished mask
                finished |= (next_token == vocab.eos_id)

                if finished.all():
                    break

    return tgt

def tokens_to_audio(generated_tokens, vocab, model_encodec,device='cuda'):
    # Convert to list and remove BOS (index 0)         first item in the batch
    tokens = generated_tokens[0].tolist()[1:]

    try:
        # Find where Speaker 2 starts
        sep_index = tokens.index(vocab.sep_id)
        s1_tokens = tokens[:sep_index]
        s2_tokens = tokens[sep_index+1:]
    except ValueError:
        print("Warning: SEP token not found in generated sequence.")  # find a solution for this later
        mid = len(tokens) // 2
        s1_tokens, s2_tokens = tokens[:mid], tokens[mid:]

    # Clean up any EOS or PAD tokens
    s1_tokens = [t for t in s1_tokens if t < vocab.vocab_size]  
    s2_tokens = [t for t in s2_tokens if t < vocab.vocab_size]

    print(f"S1 tokens count: {len(s1_tokens)}")
    print(f"S2 tokens count: {len(s2_tokens)}")

    def decode_tokens(t_list):
        if not t_list: return None
        # This reshapes 1D list into the 3D tensor EnCodec expects: [Batch Size, Number of Codebooks, Time]
        #t_tensor = torch.tensor(t_list).view(1, 1, -1).to(device)  # codebook is 1
        #focalcodec:
        t_tensor = torch.tensor(t_list).unsqueeze(0).to(device) 

        # We need to provide a dummy 'scale' list because EnCodec's decode expects it  [(t_tensor, None)]: EnCodec expects a list of "frames." Since our audio is short (3s), it fits into a single frame. The None tells EnCodec there is no external scaling factor applied.
        with torch.no_grad():
            #frames = [(t_tensor, None)]
            #wav = model_encodec.decode(frames)
            wav = model_encodec.toks_to_sig(t_tensor)
        return wav.squeeze().cpu()

    wav_s1 = decode_tokens(s1_tokens)
    wav_s2 = decode_tokens(s2_tokens)

    return wav_s1, wav_s2

#inference with beam search
def beam_search(model, src, vocab, beam_width=3, max_len=205):
    model.eval()
    device = src.device
    
    start_token = torch.tensor([[vocab.bos_id]], device=device)
    #
    beams = [(start_token, 0.0, 0.0)]  # (sequence, score, normalized_score)
    
    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            alpha = 0.7
            for seq, score, normalized_score in beams:
                if seq[0, -1].item() == vocab.eos_id:
                    new_beams.append((seq, score,normalized_score))
                    continue
                
                output = model(src, seq)
                logits = output[:, -1, :]
                probs = torch.log_softmax(logits, dim=-1) # to sum the scores instead of multiplying probabilities
                
                # 3 best tokens
                topk_probs, topk_ids = torch.topk(probs, beam_width)
                
                for i in range(beam_width):
                    next_token = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_probs[0, i].item()
                    #add length normalization
                    #length_penalty = ((5 + seq.size(1)) / 6) ** alpha                    
                    lp = ((5 + new_seq.size(1))**alpha) / ((5 + 1)**alpha)
                    normalized_score = new_score / lp
                    
                    new_beams.append((torch.cat([seq, next_token], dim=1), new_score, normalized_score))
            
            #beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            # add normalization later?
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
            
            if all(b[0][0, -1].item() == vocab.eos_id for b in beams):
                break
                
    return beams[0][0]  # Return the sequence with the highest score

def calculate_sisnr(estimated, reference):
    eps = 1e-8
    # 1. Normalize
    reference = reference - torch.mean(reference)
    estimated = estimated - torch.mean(estimated)

    # 2. Dot products for projection
    dot = torch.sum(estimated * reference)
    ref_energy = torch.sum(reference ** 2) + eps

    # 3. Scale-invariant projection
    proj = (dot / ref_energy) * reference
    noise = estimated - proj

    # 4. Ratio in Decibels (dB)
    ratio = torch.sum(proj ** 2) / (torch.sum(noise ** 2) + eps)
    sisnr = 10 * torch.log10(ratio + eps)
    return sisnr.item()

#overfitting test
def debug_overfitting(model, train_loader, optimizer, criterion, device, num_epochs=200):
    model.train()
    batch = next(iter(train_loader))
    src, tgt = batch
    src, tgt = src.to(device), tgt.to(device)
    
    tgt_input = tgt[:, :-1]
    tgt_labels = tgt[:, 1:]
    
    print("Starting Overfitting Test on a single batch...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input)
        
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_labels.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            pred = torch.argmax(output, dim=-1)
            correct = (pred == tgt_labels).sum().item()
            total = tgt_labels.numel()
            acc = (correct / total) * 100
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.4f} | Batch Acc: {acc:.2f}%")
            
            if loss.item() < 0.01:
                print("✅ Overfitting Successful! The model can memorize data.")
                break

def plot_training_history(history, save_path="training_curves.png"):

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    # --- Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'g-s', label='Validation Accuracy', markersize=4)
    plt.title('Validation Token Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Training curves saved to: {save_path}")

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience 
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, epoch, history=None, scheduler=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, history, scheduler)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, history, scheduler)
            self.counter = 0

    def save_checkpoint(self, model, optimizer, epoch, history, scheduler):
        print(f'Validation loss improved. Saving best model to {self.path}...')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': self.best_loss,
            'history': history
        }, self.path)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        # For saving checkpoints
        return {
            'step': self._step,
            'warmup': self.warmup,
            'factor': self.factor,
            'model_size': self.model_size,
            'lr': self._rate,
            'optimizer_state_dict': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.warmup = state_dict['warmup']
        self.factor = state_dict['factor']
        self.model_size = state_dict['model_size']
        self._rate = state_dict['lr']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

def test_model(model, test_loader, vocab,encodec,criterion, device, save_dir, beamsearch=True):
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
    all_si_sdrs = []
    test_results = {'si_sdr': [], 'ter': []}
    avg_val_loss, total_val_loss = 0, 0
    total_real_tokens = 0
    total_correct_tokens = 0
    avg_sep_acc, avg_val_acc = 0,-0
    total_sep_correct = 0
    total_sep_expected = 0
    

    print("\n--- Final Testing on TestSet ---")
    with torch.no_grad():
            for batch in test_loader:
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                
                tgt_input, tgt_labels = tgt[:, :-1], tgt[:, 1:]
                output = model(src, tgt_input) 
                
                # Loss
                #v_loss = criterion(output.reshape(-1, output.size(-1)), tgt_labels.reshape(-1))
                # PIT loss
                sep_idx = (tgt_labels[0] == vocab.sep_id).nonzero(as_tuple=True)[0].item()
                tgt_s1 = tgt_labels[:, :sep_idx]
                tgt_s2 = tgt_labels[:, sep_idx+1:-1] #eos
                #sep_idx = (output[0] == vocab.sep_id).nonzero(as_tuple=True)[0].item()
                output_s1 = output[:, :sep_idx, :] 
                output_s2 = output[:, sep_idx+1:-1, :]
                loss1 = criterion(output_s1.reshape(-1, output.size(-1)), tgt_s1.reshape(-1)) + \
                criterion(output_s2.reshape(-1, output.size(-1)), tgt_s2.reshape(-1))
                loss2 = criterion(output_s1.reshape(-1, output.size(-1)), tgt_s2.reshape(-1)) + \
                criterion(output_s2.reshape(-1, output.size(-1)), tgt_s1.reshape(-1))
                v_loss = torch.min(loss1, loss2)  # PIT: choose the permutation with lower loss
                sep_loss = criterion(output[:, sep_idx, :], tgt_labels[:, sep_idx])
                eos_loss = criterion(output[:, -1, :], tgt_labels[:, -1])
                v_loss += (sep_loss + eos_loss)* 0.5  # give less weight to SEP token
                total_val_loss += v_loss.item()
                
                # Token Accuracy (Ignoring Padding)
                predicted_indices = torch.argmax(output, dim=-1)
                mask = (tgt_labels != vocab.pad_id)
                correct = (predicted_indices == tgt_labels) & mask
                total_correct_tokens += correct.sum().item()
                total_real_tokens += mask.sum().item()
                # see if SEP token is predicted 
                true_sep_mask = (tgt_labels == vocab.sep_id) # ground- truth Sep positions
                num_true_sep = true_sep_mask.sum().item()
                correct_sep = (predicted_indices[true_sep_mask] == vocab.sep_id).sum().item() 
                total_sep_correct += correct_sep   
                total_sep_expected += num_true_sep 
            
    avg_val_loss = total_val_loss / len(test_loader)
    avg_val_acc = (total_correct_tokens / total_real_tokens) * 100 
    avg_sep_acc = total_sep_correct / total_sep_expected * 100
    # SI-SDR
    # with torch.no_grad():
    #     for i, (src, tgt) in enumerate(test_loader):
    #         if i >= 10: break # Limit to 10 batches for quick testing
    #         src, tgt = src.to(device), tgt.to(device)
    
    #         pred_tokens = beam_search(model, src, vocab, beam_width=3) if beamsearch else generate_tokens(src, tgt, vocab)
            
    #         decoded_audio = tokens_to_audio(pred_tokens)
    #         decoded_audio = torchaudio.functional.resample(decoded_audio, 24000, 16000)
    #         target_audio = tokens_to_audio(tgt)
    #         target_audio = torchaudio.functional.resample(target_audio, 24000, 16000)
    #         #make sure they are the same length for SI-SDR calculation
    #         min_len = min(decoded_audio.shape[-1], target_audio.shape[-1])
    #         decoded_audio = decoded_audio[..., :min_len] #
    #         target_audio = target_audio[..., :min_len]
            
    #         # metrics:
    #         score = si_sdr_metric(decoded_audio, target_audio)
    #         all_si_sdrs.append(score)
    #         if i < 3:
    #             torchaudio.save(f"test_sample_{i}_recon.wav", decoded_audio.cpu(), 24000)
    #             torchaudio.save(f"test_sample_{i}_target.wav", target_audio.cpu(), 24000)
    # print(f"Final Test SI-SDR: {np.mean(all_si_sdrs):.2f} dB")


