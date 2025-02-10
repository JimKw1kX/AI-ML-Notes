import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("Content-length", 0))
    with open(fname, "wb") as file, tqdm(

        desc=fname,
        total=total,
        unit = "iB",
        unit_scale = True,
        unit_divisor = 1024,
    ) as bar:

      for data in resp.iter_content(chunk_size=chunk_size):
         size = file.write(data)
         bar.update(size)

hellaswags = {
   
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding('gpt2')

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    url = hellaswags[split]
    file_name = os.path.join(DATA_CACHE_DIR, f'hellaswag_{split}.jsonl')
    print(f'Donwlaoding from {url} to {file_name}')
    download_file(url,file_name)

def render_example(example):
   ctx = example['ctx']
   label = example['label']
   ending = example['endings']

   data = {
      'label' : label,
      'ctx_tokens' : None,
      'ending_tokens': [],
   }

   ctx_tokens = enc.encode(ctx)
   data['ctx_tokens'] = ctx_tokens
   tok_rows = []
   mask_rows = []


   for end in ending:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

   max_len = max(len(row) for row in tok_rows)
   tokens = torch.zeros((4, max_len), dtype=torch.long)
   mask = torch.zeros((4, max_len), dtype=torch.long)
   for i, (tok_rows, mask_rows) in enumerate(zip(tok_rows, mask_rows)):
      tokens[i, :len(tok_rows)] = torch.tensor(tok_rows)
      mask[i, :len(mask_rows)] = torch.tensor(mask_rows)

   return data, tokens, mask, label


def iterate_example(split):
   download(split)
   with open(os.path.join(DATA_CACHE_DIR, f'hellaswag_{split}.jsonl'), 'r') as f:
      for line in f:
         example = json.loads(line)
         yield example

@torch.no_grad()
def eval(model_type, device):
   torch.set_float32_matmul_precision('high') # use tf32
   model = GPT2LMHeadModel.from_pretrained(model_type)
   model.to(device)

   num_correct_norm = 0
   num_correct = 0
   num_total = 0

   for example in iterate_example('val'):
      data, tokens, mask, label = render_example(example)
      tokens = tokens.to(device)
      mask = mask.to(device)

      logits = model(tokens).logits
      # evaluate the autoregressive loss at all positions
      
      shift_logits = (logits[..., :-1, :]).contiguous()
      shift_tokens = (tokens[..., 1:]).contiguous()
    
      flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
      flat_shift_tokens = shift_tokens.view(-1)

      shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
      shift_losses = shift_losses.view(tokens.size(0), -1)

      # now get the average loss just for the completion region (where mask == 1), in each row    
      shift_mask = (mask[...,1:]).contiguous() # we must shift mask, so we start at the last prompt token
      masked_shift_losses = shift_losses * shift_mask

      sum_loss = masked_shift_losses.sum(dim=1)
      avg_loss = sum_loss / shift_mask.sum(dim=1)
        
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
      pred = sum_loss.argmin().item()
      pred_norm = avg_loss.argmin().item()

      # accum stats

      num_total +=1
      num_correct += int(pred == label)
      num_correct_norm += int(pred_norm == label)
    #   print(f'{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}')

      #debug: pretty print a few example and the losses in each case

      if num_total < 10:
        print('---')
        print(f'Context:\n {example['ctx']}')
        print(f'Endings:')
        for i, end in enumerate(example['endings']):
            print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
        print(f"predicted: {pred_norm}, actual {label}")
    
if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
   parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
   args = parser.parse_args()
   eval(args.model_type,args.device)
