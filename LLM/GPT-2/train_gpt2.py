#!/usr/bin/python3
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

# pip3 install tiktoken tqdm transformers
import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from helloswag import render_example,iterate_example
# https://github.com/karpathy/build-nanogpt
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
# DDP disctriburted data parallel
from torch.nn.parallel.distributed import dist

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1    
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)

        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y   = y.transpose(1,2).contiguous().view(B,T,C)
        # y = F.scaled_dot_product_attention(q, k,v,is_causal=True)
        # y = y.transpose(1,2).contiguous().view(B,T,C)
        # y = self.c_proj(y)
        y   = self.c_proj(y) 
        return y 

 

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
 
 
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self,x ):
        x  = x + self.attn(self.ln_1(x)) # 
        x  = x + self.mlp(self.ln_2(x))
        return x


#-------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int  = 12
    n_embd: int = 768

#-------------------

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # you have to use h, else will error: KeyError: 'transformer.h.0.ln_1.weight'
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # sharing weights 
        # https://arxiv.org/pdf/1608.05859 
        # https://github.com/openai/gpt-2/blob/master/src/model.py#L147
        
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                """
                2 comes from 2 blokcs of:

                   x  = x + self.attn(self.ln_1(x)) # 
                   x  = x + self.mlp(self.ln_2(x))
                """
                std *= (2 * self.config.n_layer) **-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx -> shape of B, T
        B,T = idx.size()
        assert T <= self.config.block_size , f"Cant forward sequnce of lenegt {T},block size is {self.config.block_size} "
        # forward tokne and pos embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # pos emb for shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # tok emb for shape(B,T,n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forwarsd the final layernorm and the classifer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size->number of possible tokens)
        loss = None
        if targets is not None:
            loss= F.cross_entropy(logits.view(-1, logits.size(-1)),targets.view(-1))
        return logits, loss

    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model  = GPT(config)
        sd     = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, lr, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        ops_group = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
            print(f"num no-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")

        # create adamw ops
        # fuses all kernels for one update instead of mutiple kernles to reduce kernel overheat
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
        optimizers = torch.optim.AdamW(ops_group, lr=lr, betas=(0.9, 0.95), eps=1e-8,fused=use_fused)
        return optimizers


# -----------

import time

num_return_sequences = 5
max_length = 30
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
print(f'Using->: {device}')
# model = GPT.from_pretrained('gpt2')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# get batch data
import tiktoken

import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt


class DataloaderLite:
    
    def __init__(self, B,T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        
        assert len(shards) > 0, f'no shards found for split {split}'
        if master_process:
            print(f'found {len(shards)} shards for split {split}')
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

          
    def next_batch(self):
        B, T = self.B, self.T
        buf  = self.tokens[self.current_position:self.current_position + B* T + 1]
        # buf = buf.to(device)
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B * T * self.num_processes
        # print(f'current_position: {self.current_position}')

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# y = buf[1:].view(B,T)

#------------------------------------------


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now l thin we need cuda for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda : {ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # logging , checkpointing
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = 'mps'
        print(f'using devive: {device}') 
#------------------------------------------
# we need to put B = 0.5 M paramters but our GPU is samll, so below code solves this problem by using gardient accumulation
# row x is (B,T) B=5, T=8
device_type = 'cuda' if device.startswith('cuda') else 'cpu'
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 # 2**19, 0.5M number of tokens
B = 32 # micro batch size
T = 1024 # GPT2 1024 sequence length
# (B * T) = 16384 per forward and backward
assert total_batch_size % (B * T * ddp_world_size)  == 0, 'Make sure total_batch_size is devisible by B * T * ddp_world_size'
gard_accum_steps = total_batch_size // (B * T * ddp_world_size)
# gard_accum_steps = 5
if master_process:
    print(f"total desiered batch size : {total_batch_size}")
    print(f"=> calculated gardient accumulation steps: {gard_accum_steps}")

#------------------------------------------
train_loader = DataloaderLite(B=B,T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataloaderLite(B=B,T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')
checkpoint_path = 'log/model_19000.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu') # map_location='cpu' avoids GPU memory exhustion
# create model
model = GPT(GPTConfig(vocab_size=50304)) # defualt config using 124M paramters
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the 'raw' unwrapped model
raw_model.load_state_dict(checkpoint['model'])
max_lr = 6e-4
min_lr = max_lr * 0.1

# warmup_steps = 5
# max_steps = 5
# detect_step = 5

warmup_steps = 715
saved_step = 19000
max_steps = 19073 - saved_step # saved previous step and the model and continue from here to train 
detect_step = 500

print(f'warmup_steps {warmup_steps}')
print(f'max_steps {max_steps}')
print(f'detect_step {detect_step}')

# testing on a signle batch and its overfitting well,so next needs to create a data loader to load all the batches
# ops = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # gpt3 hyper params

def get_most_likely_row(tokens,mask, logits):
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
      pred_norm = avg_loss.argmin().item()
      return pred_norm

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0 
    return min_lr + coeff * (max_lr - min_lr) 

ops = raw_model.configure_optimizers(weight_decay=0.1, lr=6e-4, device=device_type)
enc = tiktoken.get_encoding('gpt2')

if __name__ == "__main__":
    
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log.txt')
    with open(log_file, 'w') as f:
        pass


    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps -1)

        # val loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()

            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20

                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            # if master_process:
            print(f'validation loss: {val_loss_accum.item():.4f}')
            with open(log_file, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():.4f}\n')

            if (step > 0 and step % detect_step == 0) or last_step:
                saved_step += step
                # optionally write model checkpoints
                # saving the model after saved_step in case you need to stop the training
                checkpoint_path  = os.path.join(log_dir, f'model_{saved_step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer.state_dict': ops.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f'------>: model saved to {checkpoint_path} at {saved_step}')


        # hellaswag eval
        if (step % 250 == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_example('val')):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens,mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total

            if master_process:
                print(f'HellaSwag accuray: {num_correct_norm} / {num_total}={acc_norm:.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{step} hella {acc_norm:.4f}\n')

        #  from the model (except step 0, which is noise)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("This is how Tesla's FSD and Autopilot system work: ")
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
                
            for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f'rank {ddp_rank} sample {i}: {decoded}')
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1


        model.train()
        ops.zero_grad()
        loss_accum = 0.0
        for micro_step in range(gard_accum_steps):

            x, y = train_loader.next_batch()
            x,y = x.to(device), y.to(device) 
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits,loss = model(x,y) 
            loss = loss / gard_accum_steps
            loss_accum += loss.detach() # detach tensor from graph
          

            if ddp:
                model.require_backward_grad_sync = (micro_step == gard_accum_steps -1)
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in ops.param_groups:
            param_group['lr'] = lr
        ops.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 -t0
        tokens_proccsed = train_loader.B * train_loader.T * gard_accum_steps * ddp_world_size
        token_per_sec = tokens_proccsed / dt
        if master_process:
            print(f'step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {token_per_sec:.2f}')
            with open(log_file, 'a') as f:
                f.write(f'{step} train {loss_accum.item():.6f}\n')

        # prefix tokens

    if ddp:
        destroy_process_group()


