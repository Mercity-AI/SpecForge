# Usage: 
#   --target-model-path Qwen/Qwen2.5-3B-Instruct \
#   --train-data-path ./cache/dataset/ultrachat_train.jsonl \
#   --output-dir ./outputs/eagle3_run \
#   --batch-size 1 --max-length 512

import argparse
import hashlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from specforge import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    TargetHead,
    get_eagle3_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    rank_0_priority,
)

# Training script started

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_gpu_memory_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9

def parse_args() -> Tuple[ArgumentParser, Namespace]:
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--draft-model-config", type=str, required=False)
    model_group.add_argument("--embedding-key", type=str, default="model.embed_tokens.weight")
    model_group.add_argument("--lm-head-key", type=str, default="lm_head.weight")
    model_group.add_argument("--is-vlm", action="store_true")
    model_group.add_argument("--target-model-backend", type=str, default="sglang", choices=["sglang", "hf", "custom"])

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--train-hidden-states-path", type=str, default=None)
    dataset_group.add_argument("--eval-hidden-states-path", type=str, default=None)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="llama3")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=8)

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument("--max-num-steps", type=int, default=None)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-4)
    training_group.add_argument("--max-length", type=int, default=2048)
    training_group.add_argument("--warmup-ratio", type=float, default=0.015)
    training_group.add_argument("--total-steps", type=int, default=None)
    training_group.add_argument("--max-grad-norm", type=float, default=0.5)
    training_group.add_argument("--ttt-length", type=int, default=7)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--ckpt-dir", type=str, default=None)
    training_group.add_argument("--eval-interval", type=int, default=5000)
    training_group.add_argument("--save-interval", type=int, default=5000)
    training_group.add_argument("--log-interval", type=int, default=50)
    training_group.add_argument("--log-samples-interval", type=int, default=200, help="Log detailed samples every N steps")
    training_group.add_argument("--num-samples-to-log", type=int, default=5, help="Number of samples to log in detail")
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument("--draft-accumulation-steps", type=int, default=1)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)
    optimization_group.add_argument("--attention-backend", type=str, default="flex_attention")

    other_group = parser.add_argument_group("others")
    other_group.add_argument("--cache-key", type=str, default=None)
    other_group.add_argument("--cache-dir", type=str, default="./cache")
    other_group.add_argument("--output-dir", type=str, required=True)
    other_group.add_argument("--verbose", action="store_true")
    other_group.add_argument("--dist-timeout", type=int, default=20)
    other_group.add_argument("--model-download-dir", type=str, default=None)

    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument("--min-pixels", type=int, default=50176)
    vlm_group.add_argument("--max-pixels", type=int, default=802816)

    profiling_group = parser.add_argument_group("profiling")
    profiling_group.add_argument("--profile", action="store_true")
    profiling_group.add_argument("--profile-start-step", type=int, default=30)
    profiling_group.add_argument("--profile-num-steps", type=int, default=4)
    profiling_group.add_argument("--profile-record-shapes", action="store_true")

    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig, is_online: bool = True
) -> Tuple[Union[Eagle3TargetModel, TargetHead], Optional[AutoProcessor]]:
    if is_online:
        if (
            args.is_vlm
            and draft_model_config.target_model_type == "qwen2_5_vl"
            and args.tp_size == 1
        ):
            from transformers import Qwen2_5_VLForConditionalGeneration
            target_model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .cuda()
            )
        else:
            if args.target_model_backend == "sglang":
                target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
            else:
                target_model_kwargs = {}
            target_model = get_eagle3_target_model(
                pretrained_model_name_or_path=args.target_model_path,
                backend=args.target_model_backend,
                torch_dtype=torch.bfloat16,
                device="cuda",
                cache_dir=args.model_download_dir,
                **target_model_kwargs,
            )

        if (
            hasattr(draft_model_config, "eagle_config")
            and draft_model_config.eagle_config is not None
            and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
        ):
            target_model.set_aux_hidden_states_layers(
                draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
            )
        else:
            target_model.set_aux_hidden_states_layers()

        if args.is_vlm:
            processor = AutoProcessor.from_pretrained(
                args.target_model_path,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        else:
            processor = None

        return target_model, processor
    else:
        target_head = TargetHead.from_pretrained(
            model_path=args.target_model_path,
            lm_head_key=args.lm_head_key,
            cache_dir=args.model_download_dir,
        )
        return target_head, None


def sanity_check(args: Namespace) -> None:
    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size


def build_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, nn.Module]:
    if args.draft_model_config is None:
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.model_download_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    draft_model_last_checkpoint = None
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_config = os.path.join(args.ckpt_dir, "config.json")
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Finetuning from base model: {draft_model_last_checkpoint}")
        else:
            raise ValueError(f"Provided base model dir {args.ckpt_dir} is not a valid directory.")

    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        if draft_model_last_checkpoint:
            print_on_rank0(f"Resuming from checkpoint: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()

    total_params, trainable_params = count_parameters(draft_model)
    print_on_rank0(f"Draft model: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable), GPU: {get_gpu_memory_gb():.1f}GB")

    return draft_model_config, draft_model


def build_dataloaders(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
    processor: Optional[AutoProcessor] = None,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )

        if args.train_hidden_states_path is not None:
            train_eagle3_dataset = build_offline_eagle3_dataset(
                args.train_hidden_states_path,
                args.max_length,
            )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
    )

    if args.eval_data_path is not None or args.eval_hidden_states_path is not None:
        if args.eval_data_path is not None:
            eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                is_vlm=args.is_vlm,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
                is_preformatted=args.is_preformatted,
            )
        elif args.eval_hidden_states_path is not None:
            eval_eagle3_dataset = build_offline_eagle3_dataset(
                args.eval_hidden_states_path,
                args.max_length,
            )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=args.is_vlm,
        )
    else:
        eval_dataloader = None

    return train_dataloader, vocab_mapping_path, eval_dataloader


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    eagle3_model: nn.Module,
    optimizer: Optimizer,
):
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = eagle3_model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
        }
        state_to_save.update(optimizer.state_dict())
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            eagle3_model.draft_model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
        dist.barrier()


def log_detailed_predictions(
    args: Namespace,
    eagle3_model: nn.Module,
    data: dict,
    target_model: Optional[Eagle3TargetModel],
    tokenizer: AutoTokenizer,
    global_step: int,
    tracker: Tracker,
    is_online: bool = True,
) -> None:
    """Log detailed predictions to understand drafter improvement"""
    if wandb is None:
        print_on_rank0("wandb not available, skipping detailed predictions logging")
        return
    
    # Get predictions with detailed outputs
    eagle3_model.eval()
    draft_model = eagle3_model.draft_model if hasattr(eagle3_model, 'draft_model') else eagle3_model.module.draft_model
    
    with torch.no_grad():
        if args.is_vlm:
            # For VLM, we'll skip detailed logging for now
            return
        
        if is_online:
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )
            
            input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
            attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
            loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
            target = get_dp_data_shard_from_tp(eagle3_data.target)
            hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)
        else:
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            hidden_states = data["hidden_state"].cuda()
            target = target_model(data["target"].cuda())
            input_ids, target, loss_mask = target_model.preprocess(
                input_ids, target, loss_mask
            )
        
        # Get draft model predictions using the correct API
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project hidden states
        projected_hidden_states = draft_model.project_hidden_states(hidden_states)
        
        # Prepare position IDs
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        decoder_attention_mask = draft_model.prepare_decoder_attention_mask(
            attention_mask, projected_hidden_states, batch_size, seq_length, 0
        )
        
        # Run multiple TTT steps to get predictions
        draft_logits_list = []
        hidden_states_step = projected_hidden_states
        cache_hidden = [[], []] if args.ttt_length > 1 else None
        
        for ttt_idx in range(args.ttt_length):
            # Embed input IDs
            inputs_embeds = draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states_step.dtype)
            
            # Run backbone (use_cache=False for logging since we don't need KV cache)
            hidden_states_out = draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states_step,
                cache_hidden=cache_hidden,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            
            # Compute logits
            logits = draft_model.compute_logits(hidden_states_out)
            draft_logits_list.append(logits)
            
            # Update hidden states for next iteration
            hidden_states_step = hidden_states_out
        
        # Stack logits: [ttt_length, batch, seq_len, vocab_size] -> [batch, ttt_length, seq_len, vocab_size]
        draft_logits = torch.stack(draft_logits_list, dim=1)
        draft_probs = torch.softmax(draft_logits, dim=-1)  # Convert to probabilities
        draft_predictions = torch.argmax(draft_logits, dim=-1)  # [batch, ttt_length, seq_len]
        
        # Prepare logging data
        num_samples = min(args.num_samples_to_log, input_ids.shape[0])
        log_data = []
        
        for sample_idx in range(num_samples):
            sample_input_ids = input_ids[sample_idx]
            sample_loss_mask = loss_mask[sample_idx]
            sample_target = target[sample_idx]
            sample_draft_preds = draft_predictions[sample_idx]
            
            # Find non-padded positions
            valid_positions = sample_loss_mask.nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            
            # Get a window of tokens to log (not the entire sequence)
            start_pos = max(0, valid_positions[0].item() - 10)
            end_pos = min(len(sample_input_ids), valid_positions[-1].item() + 10)
            
            # Decode input context
            context_tokens = sample_input_ids[start_pos:end_pos]
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)
            
            # Log predictions for each draft position
            position_logs = []
            for ttt_idx in range(min(args.ttt_length, draft_predictions.shape[1])):
                # Get predictions at this TTT position
                ttt_preds = sample_draft_preds[ttt_idx]
                ttt_targets = sample_target[ttt_idx] if target.dim() > 2 else sample_target
                ttt_probs = draft_probs[sample_idx, ttt_idx]  # Probabilities for this position
                
                # Find valid prediction positions
                valid_pred_positions = valid_positions[valid_positions < len(ttt_preds)]
                if len(valid_pred_positions) == 0:
                    continue
                
                # Take a few positions to log
                positions_to_show = valid_pred_positions[:min(20, len(valid_pred_positions))]
                
                pred_tokens = ttt_preds[positions_to_show]
                target_tokens = ttt_targets[positions_to_show]
                
                # Get confidence scores
                pred_confidences = []
                target_confidences = []
                for pos_idx, pos in enumerate(positions_to_show):
                    pred_token = pred_tokens[pos_idx].item()
                    target_token = target_tokens[pos_idx].item()
                    
                    # Get probability of predicted token
                    pred_conf = ttt_probs[pos, pred_token].item()
                    pred_confidences.append(pred_conf)
                    
                    # Get probability assigned to target token
                    target_conf = ttt_probs[pos, target_token].item()
                    target_confidences.append(target_conf)
                
                avg_pred_conf = sum(pred_confidences) / len(pred_confidences) if pred_confidences else 0
                avg_target_conf = sum(target_confidences) / len(target_confidences) if target_confidences else 0
                
                # Calculate accuracy
                correct = (pred_tokens == target_tokens).sum().item()
                total = len(positions_to_show)
                accuracy = correct / total if total > 0 else 0
                
                # Decode predictions and targets
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
                target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
                
                position_logs.append({
                    'position': ttt_idx,
                    'accuracy': f"{accuracy:.2%}",
                    'predicted': pred_text,
                    'target': target_text,
                    'match': '✓' if pred_text == target_text else '✗',
                    'pred_confidence': f"{avg_pred_conf:.3f}",
                    'target_confidence': f"{avg_target_conf:.3f}"
                })
            
            log_data.append({
                'sample': sample_idx,
                'context': context_text[:200] + '...' if len(context_text) > 200 else context_text,
                'predictions': position_logs
            })
        
        # Log to wandb
        if dist.get_rank() == 0 and tracker.is_initialized:
            # Log model internals
            print_on_rank0(f"\n[Model Internals at step {global_step}]")
            print_on_rank0(f"  Input shape: {input_ids.shape}")
            print_on_rank0(f"  Hidden states shape: {hidden_states.shape}")
            print_on_rank0(f"  Target shape: {target.shape}")
            print_on_rank0(f"  Loss mask sum: {loss_mask.sum().item()}")
            print_on_rank0(f"  Draft predictions shape: {draft_predictions.shape}")
            
            # Log hidden states statistics
            hidden_stats = {
                'hidden_states/mean': hidden_states.mean().item(),
                'hidden_states/std': hidden_states.std().item(),
                'hidden_states/min': hidden_states.min().item(),
                'hidden_states/max': hidden_states.max().item(),
            }
            tracker.log(hidden_stats, step=global_step)
            
            # Create a formatted table
            table_data = []
            confidence_data = []
            for sample in log_data:
                for pred_log in sample['predictions']:
                    table_data.append([
                        global_step,
                        sample['sample'],
                        sample['context'][:100],
                        pred_log['position'],
                        pred_log['accuracy'],
                        pred_log['predicted'][:100],  # Truncate for readability
                        pred_log['target'][:100],
                        pred_log['match'],
                        pred_log['pred_confidence'],
                        pred_log['target_confidence']
                    ])
                    confidence_data.append({
                        'position': pred_log['position'],
                        'pred_conf': float(pred_log['pred_confidence']),
                        'target_conf': float(pred_log['target_confidence'])
                    })
            
            if table_data:
                table = wandb.Table(
                    columns=['step', 'sample', 'context', 'draft_position', 'accuracy', 
                            'predicted', 'target', 'match', 'pred_conf', 'target_conf'],
                    data=table_data
                )
                tracker.log({'predictions/samples_table': table}, step=global_step)
                
                # Log average confidence by position
                for pos in range(args.ttt_length):
                    pos_confs = [c for c in confidence_data if c['position'] == pos]
                    if pos_confs:
                        avg_pred_conf = sum(c['pred_conf'] for c in pos_confs) / len(pos_confs)
                        avg_target_conf = sum(c['target_conf'] for c in pos_confs) / len(pos_confs)
                        tracker.log({
                            f'predictions/pos_{pos}_pred_confidence': avg_pred_conf,
                            f'predictions/pos_{pos}_target_confidence': avg_target_conf,
                        }, step=global_step)
                
                # Also log as text for easy reading
                text_log = f"\n{'='*80}\nStep {global_step} - Detailed Predictions\n{'='*80}\n"
                text_log += f"\nModel Flow:\n"
                text_log += f"  1. Target model generates hidden states from input_ids\n"
                text_log += f"  2. Draft model receives: input_ids + hidden_states\n"
                text_log += f"  3. Draft model predicts next tokens at {args.ttt_length} positions\n"
                text_log += f"  4. Predictions compared against target model's outputs\n\n"
                
                for sample in log_data:
                    text_log += f"\n[Sample {sample['sample']}]\n"
                    text_log += f"Context: {sample['context']}\n"
                    for pred_log in sample['predictions']:
                        text_log += f"\n  Position {pred_log['position']} (Acc: {pred_log['accuracy']}) {pred_log['match']}\n"
                        text_log += f"    Predicted: {pred_log['predicted']} (conf: {pred_log['pred_confidence']})\n"
                        text_log += f"    Target:    {pred_log['target']} (conf: {pred_log['target_confidence']})\n"
                        
                        # Highlight if model is confident but wrong, or uncertain but right
                        pred_conf_val = float(pred_log['pred_confidence'])
                        target_conf_val = float(pred_log['target_confidence'])
                        if pred_log['match'] == '✗' and pred_conf_val > 0.7:
                            text_log += f"    ⚠️  High confidence in wrong prediction!\n"
                        elif pred_log['match'] == '✓' and target_conf_val < 0.3:
                            text_log += f"    ⚠️  Low confidence in correct prediction\n"
                
                print_on_rank0(text_log)
                tracker.log({'predictions/detailed_text': wandb.Html(f"<pre>{text_log}</pre>")}, step=global_step)
                
                # Create a summary chart showing improvement over time
                if confidence_data:
                    overall_pred_conf = sum(c['pred_conf'] for c in confidence_data) / len(confidence_data)
                    overall_target_conf = sum(c['target_conf'] for c in confidence_data) / len(confidence_data)
                    tracker.log({
                        'predictions/overall_pred_confidence': overall_pred_conf,
                        'predictions/overall_target_confidence': overall_target_conf,
                        'predictions/confidence_gap': overall_target_conf - overall_pred_conf,
                    }, step=global_step)
                    
                    print_on_rank0(f"\n[Confidence Summary]")
                    print_on_rank0(f"  Avg confidence in predictions: {overall_pred_conf:.3f}")
                    print_on_rank0(f"  Avg confidence in targets: {overall_target_conf:.3f}")
                    print_on_rank0(f"  Gap (target - pred): {overall_target_conf - overall_pred_conf:.3f}")
                    print_on_rank0(f"  → {'Model improving!' if overall_target_conf > overall_pred_conf else 'Model needs more training'}\n")
    
    eagle3_model.train()


def run_forward(
    args: Namespace,
    eagle3_model: nn.Module,
    data: dict,
    target_model: Optional[Eagle3TargetModel] = None,
    is_online: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if args.is_vlm:
        plosses, _, acces = eagle3_model(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
            pixel_values=data["pixel_values"].cuda(),
            image_grid_thw=data["image_grid_thw"].cuda(),
        )
    else:
        if is_online:
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
            attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
            loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
            target = get_dp_data_shard_from_tp(eagle3_data.target)
            hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)
        else:
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            hidden_states = data["hidden_state"].cuda()
            target = target_model(data["target"].cuda())
            input_ids, target, loss_mask = target_model.preprocess(
                input_ids, target, loss_mask
            )

        plosses, _, acces = eagle3_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target=target,
            hidden_states=hidden_states,
        )
    return plosses, acces


def run_backward_and_update(
    args: Namespace, plosses: List[torch.Tensor], optimizer: Optimizer, global_step: int
) -> None:
    ploss_weight = [0.8**i for i in range(len(plosses))]
    ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )
    ploss.backward()

    if global_step % args.draft_accumulation_steps == 0:
        optimizer.step()


def record_metrcs(
    args: Namespace,
    accuracies: List[torch.Tensor],
    plosses: List[torch.Tensor],
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    # Stack tensors directly (they should already be detached or we'll detach them)
    # Each tensor in the list should be a scalar (0-dimensional) tensor
    accuracies_tensor = torch.stack([acc.detach() if isinstance(acc, torch.Tensor) else torch.tensor(acc, device="cuda") for acc in accuracies])
    plosses_tensor = torch.stack([pl.detach() if isinstance(pl, torch.Tensor) else torch.tensor(pl, device="cuda") for pl in plosses])
    
    # Ensure tensors are on CUDA for all_reduce
    if accuracies_tensor.device.type != "cuda":
        accuracies_tensor = accuracies_tensor.cuda()
    if plosses_tensor.device.type != "cuda":
        plosses_tensor = plosses_tensor.cuda()
    
    assert accuracies_tensor.shape[0] == args.ttt_length
    
    # Debug: Check values BEFORE all_reduce
    if args.verbose and dist.get_rank() == 0:
        print_on_rank0(f"DEBUG: Before all_reduce - accuracies: {accuracies_tensor.cpu().tolist()}, plosses: {plosses_tensor.cpu().tolist()}")
    
    # Average across all processes
    dist.all_reduce(accuracies_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(plosses_tensor, op=dist.ReduceOp.AVG)
    
    # Debug: Check values AFTER all_reduce
    if args.verbose and dist.get_rank() == 0:
        print_on_rank0(f"DEBUG: After all_reduce - accuracies: {accuracies_tensor.cpu().tolist()}, plosses: {plosses_tensor.cpu().tolist()}")
    
    # Extract values AFTER all_reduce to get the averaged values
    accuracies_list = []
    plosses_list = []
    for i in range(args.ttt_length):
        acc_val = accuracies_tensor[i].item()
        loss_val = plosses_tensor[i].item()
        accuracies_list.append(acc_val)
        plosses_list.append(loss_val)
    
    for i in range(len(accuracies_list)):
        acc_value = float(accuracies_list[i])
        # Ensure it's a native Python float, not numpy/torch type
        acc_value = float(acc_value)
        logdict[f"{mode}/acc_{i}"] = acc_value
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i},  Acc: {acc_value:.2f}"
        )

    for i in range(len(plosses_list)):
        loss_value = float(plosses_list[i])
        # Ensure it's a native Python float, not numpy/torch type
        loss_value = float(loss_value)
        logdict[f"{mode}/ploss_{i}"] = loss_value
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i}, pLoss: {loss_value}"
        )
    
    # Tracker.log() already handles rank 0 check internally
    # Debug output (only if --verbose flag is set)
    if args.verbose and dist.get_rank() == 0:
        print_on_rank0(f"DEBUG: Logdict values: {[(k, v) for k, v in logdict.items() if 'acc' in k or 'ploss' in k]}")
        print_on_rank0(f"DEBUG: Logdict types: {[(k, type(v).__name__) for k, v in logdict.items() if 'acc' in k or 'ploss' in k]}")
        print_on_rank0(f"DEBUG: Tracker type: {type(tracker).__name__}, initialized: {tracker.is_initialized}, rank: {tracker.rank}")
        # Verify values are not zero
        non_zero_values = [(k, v) for k, v in logdict.items() if ('acc' in k or 'ploss' in k) and v != 0.0]
        if len(non_zero_values) == 0:
            print_on_rank0("WARNING: All logged values are zero!")
        else:
            print_on_rank0(f"DEBUG: Non-zero values: {non_zero_values[:3]}...")  # Show first 3
    
    tracker.log(logdict, step=global_step)


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    is_online = (
        args.train_data_path is not None and args.train_hidden_states_path is None
    )
    sanity_check(args)
    
    print_on_rank0(f"Training: {args.target_model_path} -> {args.output_dir}")
    print_on_rank0(f"Mode: {'online' if is_online else 'offline'}, World size: {dist.get_world_size()}, TP: {args.tp_size}, DP: {args.dp_size}")

    # Build models
    draft_model_config, draft_model = build_draft_model(args)
    target_model, processor = build_target_model(args, draft_model_config, is_online)

    # Build dataloaders and tokenizer for logging
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config, processor
    )
    draft_model.load_vocab_mapping(vocab_mapping_path)
    
    # Keep tokenizer for detailed logging
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    
    print_on_rank0(f"Data: {len(train_dataloader.dataset):,} samples, {len(train_dataloader)} steps/epoch")
    if eval_dataloader is not None:
        print_on_rank0(f"Eval: {len(eval_dataloader)} batches")

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch

    # Build Eagle3 model
    if (
        args.is_vlm
        and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl"
    ):
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        eagle3_model = OnlineEagle3Model(
            draft_model=draft_model,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )

    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,
    )

    # Build optimizer and tracker
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    tracker = build_tracker(args, parser)
    
    print_on_rank0(f"Config: TTT={args.ttt_length}, backend={args.attention_backend}, lr={args.learning_rate}, steps={args.total_steps}")
    
    global_step = 0
    start_epoch = 0
    dist.barrier()

    last_time = time.time()
    
    # Store a fixed batch for consistent detailed logging
    logging_batch = None

    # Start training
    print_on_rank0(f"\nStarting training: {args.num_epochs} epochs, eval every {args.eval_interval} steps, save every {args.save_interval} steps\n")
    print_on_rank0(f"Detailed prediction logging: every {args.log_samples_interval} steps, {args.num_samples_to_log} samples\n")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            global_step += 1
            
            # Store first batch for consistent detailed logging
            if logging_batch is None and not args.is_vlm:
                logging_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                print_on_rank0(f"Captured logging batch with {logging_batch['input_ids'].shape[0]} samples")

            # Check for empty loss masks
            if "loss_mask" in data:
                mask_sum = data["loss_mask"].sum().item()
                if mask_sum == 0:
                    print_on_rank0(f"WARNING: Empty loss_mask at step {global_step}! The model is not learning anything. Check your chat template and data format.")
                elif args.verbose and global_step % args.log_interval == 0:
                    print_on_rank0(f"DEBUG: loss_mask sum = {mask_sum}")
            
            # ================================================
            # Profiling
            # ================================================
            if args.profile:
                if global_step == args.profile_start_step + 1:
                    print("Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if global_step == args.profile_start_step + args.profile_num_steps + 1:
                    output_path = os.path.join(
                        args.output_dir,
                        f"profile_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            # ================================================
            # Training Step
            # ================================================
            plosses, acces = run_forward(
                args, eagle3_model, data, target_model, is_online
            )
            run_backward_and_update(args, plosses, optimizer, global_step)

            # Log training metrics
            if global_step % args.log_interval == 0:
                record_metrcs(
                    args, acces, plosses, global_step, tracker, optimizer, mode="train"
                )
            
            # Log detailed predictions periodically
            if global_step % args.log_samples_interval == 0 and logging_batch is not None and not args.is_vlm:
                print_on_rank0(f"\n{'='*80}\nLogging detailed predictions at step {global_step}\n{'='*80}")
                log_detailed_predictions(
                    args, eagle3_model, logging_batch, target_model, tokenizer, global_step, tracker, is_online
                )

            time_per_step = time.time() - last_time
            last_time = time.time()
            avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
            avg_acc = sum(acc.item() for acc in acces) / len(acces)

            if global_step % args.log_interval == 0:
                print_on_rank0(
                    f"[{global_step:>6}/{args.total_steps}] "
                    f"loss={avg_loss:.4f} acc={avg_acc:.1%} "
                    f"lr={optimizer.get_learning_rate():.2e} "
                    f"t={time_per_step:.1f}s gpu={get_gpu_memory_gb():.1f}GB"
                )
                # Per-position losses
                loss_str = " ".join([f"L{i}={plosses[i].item():.4f}" for i in range(len(plosses))])
                acc_str = " ".join([f"A{i}={acces[i].item():.1%}" for i in range(len(acces))])
                print_on_rank0(f"  Losses: {loss_str}")
                print_on_rank0(f"  Accs:   {acc_str}")

            if dist.get_rank() == 0:
                progress_bar.set_postfix({"loss": f"{avg_loss:.3f}", "acc": f"{avg_acc:.1%}"})

            # Evaluation
            if (
                args.eval_data_path is not None
                and global_step % args.eval_interval == 0
            ):
                print_on_rank0(f"\n[Eval @ step {global_step}]")
                draft_model.eval()
                eval_acces = [[] for _ in range(eagle3_model.length)]
                eval_plosses = [[] for _ in range(eagle3_model.length)]

                for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                    with torch.no_grad():
                        plosses, acces = run_forward(
                            args, eagle3_model, data, target_model, is_online
                        )
                        eval_acces = [
                            eval_acces[i] + [acces[i]] for i in range(len(acces))
                        ]
                        eval_plosses = [
                            eval_plosses[i] + [plosses[i]] for i in range(len(plosses))
                        ]

                eval_acces = [torch.stack(acc).mean() for acc in eval_acces]
                eval_plosses = [torch.stack(pl).mean() for pl in eval_plosses]

                record_metrcs(
                    args,
                    eval_acces,
                    eval_plosses,
                    global_step,
                    tracker,
                    mode="eval",
                )
                
                # Log eval summary
                eval_avg_loss = sum(pl.item() for pl in eval_plosses) / len(eval_plosses)
                eval_avg_acc = sum(acc.item() for acc in eval_acces) / len(eval_acces)
                print_on_rank0(f"  Eval avg: loss={eval_avg_loss:.4f} acc={eval_avg_acc:.1%}")

            # Save checkpoints
            if global_step % args.save_interval == 0:
                print_on_rank0(f"\n[Checkpoint @ step {global_step}]")
                save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        # End-of-epoch checkpoint
        print_on_rank0(f"\n[End of Epoch {epoch} - Checkpoint @ step {global_step}]")
        save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break

    # Cleanup
    print_on_rank0(f"\nTraining completed: {global_step} steps")
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()