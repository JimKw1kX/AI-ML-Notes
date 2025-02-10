#!/usr/bin/python3
import torch
import sys,time,os
import torch.nn.functional as F
from train_gpt2 import device, model,enc,ddp_rank,device_type,GPTConfig  
# Initialize the model
def run():
    # Load the model's state dictionary
    state = torch.load('\path_tosaved_model\model_19072.pt', map_location='cpu')
    # state = torch.load('/home/ubuntu/GPT2/soph/seed0/epoch_1.pt', map_location=device)
    model_state_dict = state['model']
    model.load_state_dict(model_state_dict)
    loss_value = state['val_loss']  # Extract the loss value
    epoch = state.get('step')
    print(f"Last loss value from training: {loss_value:.4f}")
    print(f"model was saved after iteration: {epoch}")

    # Set model to evaluation mode
    model.eval()

# Generate text
 
    num_return_sequences = 50
    max_length = 500
    tokens = enc.encode("This is how Tesla FSD works, ")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):

                logits, loss = model(xgen) # (B,T,vocab_size)
                # take the logits at the last position
                logits = logits[:, -1,:] # (B, vocab_size)
                # get probailities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B,1)
                xcol = torch.gather(topk_indices,-1,ix) # (B,1)
                # append tp the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

    # with open('tesla2.txt', 'a') as f:    
    for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
        
            for char in decoded:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.01)
            #  print character by character
            text = '\n' + '-' * 80 + '\n'
            sys.stdout.write(text)
            sys.stdout.flush()


if __name__ == "__main__":
    run()