import torch
import math
import os
import torch.nn.functional as F

def get_optimizer(
    model,
    device_type,
    learning_rate = 5e-4,  # max learning rate
    weight_decay = 1e-1,
    beta1 = 0.9,
    beta2 = 0.95,
):
    
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), fused=(device_type=='cuda'))
    return optimizer


def compute_loss(logits, targets):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1), 
        ignore_index=-1
    )
    return loss



# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, iter_batches, eval_iters, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split, seed=10)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X)
                loss = compute_loss(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def generate_paragraph(
        model, 
        prompt, 
        tokenizer,
        device='cuda:0',
        temperature=0.5, 
        top_k=10, 
        max_new_tokens=100,
        ):
    tokenized_prompt = [tokenizer.bos_id()] + tokenizer.encode(prompt)# bos=True, eos=False)
    tokenized_prompt = (torch.tensor(tokenized_prompt, dtype=torch.long, device=device)[None, ...])
    context_tokens = tokenized_prompt
    for _ in range(max_new_tokens):
        context_tokens = context_tokens[:, -min(model.params.max_seq_len, context_tokens.size(1)):]
        output = model(context_tokens)
        
        logits = output[:, -1, :]
        
        temp_scaled_logits = logits / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context_tokens = torch.cat((context_tokens, next_token), dim=1)
    return context_tokens, tokenizer.decode(context_tokens[0].tolist())


def save_checkpoint(
        model, 
        optimizer, 
        model_args, 
        iter_num, 
        out_dir,
        out_fn='ckpt.pt'
        ):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, out_fn))


def get_lr(
        it, 
        max_iters,
        learning_rate=5e-4, 
        min_lr=0, 
        warmup_iters=1000,
        ):

    lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)