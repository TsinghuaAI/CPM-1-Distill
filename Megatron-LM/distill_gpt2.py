# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT2"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
from collections import defaultdict
from deepspeed.runtime.constants import SCHEDULER_TYPE_DEFAULT
from data.samplers import DistributedBatchSampler
from data.gpt2_dataset import build_train_valid_test_datasets
from gpt2_data_loader import make_gpt2_dataloaders
import torch.distributed as dist
from utils import print_rank_0, save_rank_0
from utils import print_params_min_max_norm
from utils import print_args
from utils import report_memory
from utils import load_checkpoint
from utils import save_checkpoint
from utils import Timers
from apex.optimizers import FusedAdam as Adam
import mpu
from model import gpt2_get_params_for_weight_decay_optimization
from model import GPT2Model
from learning_rates import AnnealingLR
from fp16 import FP16_Optimizer
from fp16 import FP16_Module
from configure_data import configure_data
from arguments import get_args
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
import numpy as np
import math
import random
import os
import json
from datetime import datetime
import time
from tqdm import tqdm
USE_TORCH_DDP = False


if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP

# torch.autograd.set_detect_anomaly(True)

def get_model_wo_parallel(args, config, do_fp16=False):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(
        **config,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers,
        parallel_output=False)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and do_fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if do_fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def get_model(args, config, do_fp16=False):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(
        **config,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers,
        parallel_output=True)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and do_fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if do_fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def get_optimizer(model, args, do_fp16=False):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = gpt2_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if do_fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})
    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler


def setup_model_and_optimizer(args, config, need_optim=False, ckpt_path=None, do_fp16=False):
    """Setup model and optimizer."""

    model = get_model(args, config, do_fp16=do_fp16)
    optimizer = get_optimizer(model, args, do_fp16=do_fp16) if need_optim else None

    lr_scheduler = get_learning_rate_scheduler(optimizer, args) if need_optim else None

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )

    iteration = 0
    if ckpt_path is not None:
        iteration = load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, args)

    return model, optimizer, lr_scheduler, iteration


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        "<eod>",
        args.reset_position_ids,
        args.reset_attention_mask)
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, student_model, teacher_model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    # Forward model.
    s_logits = student_model(tokens, position_ids,
                             attention_mask)  # [b * s * v_p]
    
    losses = {}

    s_logits = s_logits.contiguous().float()

    loss_mask = loss_mask.view(-1)
    if args.alpha_ce > 0 or args.alpha_mse > 0 or args.alpha_attn > 0 or args.alpha_hidden > 0:
        with torch.no_grad():
            t_logits = teacher_model(tokens, position_ids,
                                     attention_mask)  # [b * s * v_p]
        # NOTE: whatever precision t_logits is, convert it to float
        t_logits = t_logits.contiguous().float()
    
    if args.alpha_ce > 0:
        ce_loss = mpu.parallel_soft_cross_entropy_loss(
            s_logits / args.temperature_kd, 
            t_logits / args.temperature_kd
        ) * (args.temperature_kd) ** 2
        ce_loss = torch.sum(ce_loss.view(-1) * loss_mask) / loss_mask.sum()

        losses["ce_loss"] = ce_loss

    if args.alpha_lm > 0:        
        lm_loss = mpu.parallel_cross_entropy_loss(
            s_logits.contiguous().float(),
            labels
        )
        lm_loss = torch.sum(lm_loss.view(-1) * loss_mask) / loss_mask.sum()
        losses["lm_loss"] = lm_loss

    tot_loss = args.alpha_ce * losses.get("ce_loss", 0) + \
                args.alpha_lm * losses.get("lm_loss", 0)
    
    losses["tot_loss"] = tot_loss

    return losses


def backward_step(optimizer, model, loss, args, timers):
    """Backward step."""

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Reduce across processes.
    # lm_loss_reduced = loss

    # reduced_losses = loss.view(1)

    # if args.deepspeed:
    #     # DeepSpeed backward propagation already addressed all reduce communication.
    #     # Reset the timer to avoid breaking timer logs below.
    #     timers('allreduce').reset()
    # else:
    #     torch.distributed.all_reduce(reduced_losses.data)
    #     reduced_losses.data = reduced_losses.data / args.world_size
    #     if not USE_TORCH_DDP:
    #         timers('allreduce').start()
    #         model.allreduce_params(reduce_after=False,
    #                                fp32_allreduce=args.fp32_allreduce)
    #         timers('allreduce').stop()

    # lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return loss


def reduce_loss(args, model, timers, loss):
    reduced_losses = loss.view(1)
    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size
        if not USE_TORCH_DDP:
            timers('allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('allreduce').stop()

    return reduced_losses


def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", torch.cuda.memory_allocated() /
              (1024*1024*1024), "GigaBytes")
        print("Max Memory Allocated ",
              torch.cuda.max_memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Cache Allocated ", torch.cuda.memory_cached() /
              (1024*1024*1024), "GigaBytes")
        print("Max cache Allocated ", torch.cuda.max_memory_cached() /
              (1024*1024*1024), "GigaBytes")
        print(" ")
        #input("Press Any Key To Continue ..")


def train_step(data_iterator,
               student_model, teacher_model,
               optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    losses = forward_step(data_iterator, student_model,
                          teacher_model, args, timers)
    tot_loss = losses["tot_loss"]
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, student_model, tot_loss, args, timers)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if args.deepspeed:
        student_model.step()
    else:
        optimizer.step()

        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1
    timers('optimizer').stop()

    for k in losses:
        losses[k] = reduce_loss(args, student_model, timers, losses[k])

    return losses, skipped_iter


def train(student_model, teacher_model,
          optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator,
          timers, args):
    """Train the model."""

    # Turn on training mode which enables dropout.
    student_model.train()
    if teacher_model is not None:
        teacher_model.eval()

    # Tracking loss.
    total_losses = defaultdict(int)

    # Iterations.
    # iteration = args.iteration
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    for iter in tqdm(range(args.iteration, args.train_iters), disable=(torch.distributed.get_rank() != 0), desc="Training"):
        # while iteration < args.train_iters:

        losses, skipped_iter = train_step(train_data_iterator,
                                           student_model,
                                           teacher_model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers)
        skipped_iters += skipped_iter
        # iteration += 1

        # Update losses.
        for k in losses:
            total_losses[k] += losses[k].data.detach().float()

        # Logging.
        if iter % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            for k in total_losses:
                total_losses[k] = total_losses[k].item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            log_string = ' iteration {:8d}/{:8d} |'.format(iter,
                                                           args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3} |'.format(learning_rate)
            for k in total_losses:
                log_string += ' {} {:.6} |'.format(k, total_losses[k])
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
            
            print_rank_0(log_string)
            save_rank_0(args, log_string)
            
            for k in total_losses:
                total_losses[k] = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iter))
                report_memory_flag = False
            if USE_TORCH_DDP:
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and iter % args.save_interval == 0:
            save_checkpoint(iter, student_model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and iter % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iter)
            evaluate_and_print_results(
                prefix, val_data_iterator, student_model, teacher_model, args, timers, False)

        if args.exit_interval and iter % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iter), flush=True)
            exit()

    return iter, skipped_iters


def evaluate(data_iterator, student_model, teacher_model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    student_model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    total_losses = defaultdict(int)

    with torch.no_grad():
        for iter in tqdm(range(args.eval_iters), disable=(torch.distributed.get_rank() != 0), desc="Evaluating"):
            if verbose and iter % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iter, args.eval_iters))
                save_rank_0(args, 'Evaluating iter {}/{}'.format(iter, args.eval_iters))

            # Forward evaluation.
            losses = forward_step(data_iterator, student_model, teacher_model, args, timers)
            # tot_loss = losses["tot_loss"]
            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            if isinstance(student_model, DDP):
                for k in losses:
                    torch.distributed.all_reduce(losses[k].data)
                    losses[k].data = losses[k].data / args.world_size

            for k in losses:
                total_losses[k] += losses[k].data.detach().float().item()

    # Move model back to the train mode.
    student_model.train()

    for k in total_losses:
        total_losses[k] /= args.eval_iters
    
    return total_losses


def evaluate_and_print_results(prefix, data_iterator, student_model, teacher_model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    # TODO: do ppl for tot loss? 
    losses = evaluate(data_iterator, student_model, teacher_model, args, timers, verbose)
    lm_ppl = None
    if "lm_loss" in losses:
        lm_loss = losses["lm_loss"]
        lm_ppl = math.exp(min(20, lm_loss))
    print_rank_0('-' * 100)
    save_rank_0(args, '-' * 100)
    string = ' validation loss at {} | '.format(prefix)
    for k in losses:
        string += '{}: {:.6} | '.format(k, losses[k])
    if lm_ppl is not None:
        string += 'LM PPL: {:.6}'.format(lm_ppl)
    length = len(string) + 1
    print_rank_0('-' * length)
    save_rank_0(args, '-' * 100)
    print_rank_0(string)
    save_rank_0(args, string)
    print_rank_0('-' * length)
    save_rank_0(args, '-' * 100)

    return losses


'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''


def set_deepspeed_activation_checkpointing(args):

    deepspeed.checkpointing.configure(
        mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        if args.use_npy_data_loader:
            (train_data, val_data, test_data), num_tokens, \
                eod_token = make_gpt2_dataloaders(args)
        else:
            data_config = configure_data()
            data_config.set_defaults(data_set_type='GPT2', transpose=False)
            (train_data, val_data, test_data), tokenizer = data_config.apply(
                args)
            num_tokens = tokenizer.num_tokens
            eod_token = tokenizer.get_command('eos').Id
            assert eod_token == tokenizer.get_command('pad').Id
        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
            mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        save_rank_0(args, '> padded vocab (size: {}) with {} dummy '
             'tokens (new size: {})'.format(
                 before, after - before, after))
        print_rank_0('> found end-of-document token: {}'.format(eod_token))
        save_rank_0(args, '> found end-of-document token: {}'.format(eod_token))
        
        token_counts = torch.cuda.LongTensor([after, eod_token, int(
            args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()

    return train_data, val_data, test_data, num_tokens, eod_token


def make_data_loader(dataset):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    args = get_args()

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        print_args(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # prepare log file
    os.makedirs(args.save, exist_ok=True)
    with open(args.log_file, "w") as f:
        f.write("Logging:\n")

    # Model, optimizer, and learning rate.
    with open(args.student_config_path, "r") as f:
        student_config = json.load(f)
    
    student_model, optimizer, lr_scheduler, student_iteration = setup_model_and_optimizer(
        args, student_config, need_optim=True, ckpt_path=args.student_load, do_fp16=args.fp16)

    args.iteration = student_iteration

    teacher_model = None
    if args.teacher_config_path is not None and args.teacher_load is not None:
        with open(args.teacher_config_path, "r") as f:
            teacher_config = json.load(f)
        teacher_model, _, _, _ = setup_model_and_optimizer(
            args, teacher_config, need_optim=True, ckpt_path=args.teacher_load, do_fp16=(args.fp16 or args.teacher_fp16))

    if torch.distributed.get_rank() == 0:
        print(student_iteration)

    train_data_iterator, val_data_iterator, test_data_iterator = \
        build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider, args)

    # Resume data loader if necessary.
    # if args.resume_dataloader:
    #    if train_data is not None:
    #        train_data.batch_sampler.start_iter = args.iteration % \
    #                                              len(train_data)
    #    if val_data is not None:
    #        start_iter_val = (args.train_iters // args.save_interval) * \
    #                         args.eval_interval
    #        val_data.batch_sampler.start_iter = start_iter_val % \
    #                                            len(val_data)
    # if train_data is not None:
    #    train_data_iterator = iter(train_data)
    # else:
    #    train_data_iterator = None
    # if val_data is not None:
    #    val_data_iterator = iter(val_data)
    # else:
    #    val_data_iterator = None

    # TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.do_train:
        iteration, skipped = train(student_model, teacher_model,
                                   optimizer,
                                   lr_scheduler,
                                   train_data_iterator,
                                   val_data_iterator,
                                   timers, args)

        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, val_data_iterator,
                                              student_model, teacher_model, args, timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, student_model, optimizer, lr_scheduler, args)

    # if test_data is not None:
    #    test_data_iterator = iter(test_data)
    # else:
    #    test_data_iterator = None
    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   student_model, teacher_model, args, timers, True)


def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    save_rank_0(args, '> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")
    save_rank_0(args, "> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider, args):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.do_train
        do_valid = valid_dataloader is not None and args.do_valid
        do_test = test_dataloader is not None and args.do_test
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


if __name__ == "__main__":
    main()
