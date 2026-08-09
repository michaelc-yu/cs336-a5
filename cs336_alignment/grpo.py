import torch
from einops import rearrange
from typing import Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False, padding=False, truncation=True)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False, padding=False, truncation=True)["input_ids"]

    assert len(prompt_ids) == len(output_ids)

    input_ids = [prompt + response for prompt, response in zip(prompt_ids, output_ids)]

    # Pad the sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    for id in input_ids:
        pad_len = max_len - len(id)
        padded_ids.append(id + [tokenizer.pad_token_id] * pad_len)

    input_ids = torch.tensor(padded_ids)

    # Construct mask
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (prompt, output) in enumerate(zip(prompt_ids, output_ids)):
        len_prompt = len(prompt)
        len_output = len(output)
        response_mask[i, len_prompt : len_prompt + len_output] = 1


    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    response_mask = response_mask[:, 1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # shape (batch_sz, seq_len, vocab_size)
    token_log_probs = rearrange(
        torch.gather(log_probs, 2, rearrange(labels, "b s -> b s 1")),
        "b s 1 -> b s",
    )

    res = {'log_probs': token_log_probs}
    if return_token_entropy:
        token_entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
        res['token_entropy'] = token_entropy

    return res


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)

    all_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        # reward_dict contains keys 'format_reward', 'answer_reward', and 'reward'
        reward_dict = reward_fn(response, ground_truth)
        all_rewards.append(reward_dict)

    raw_rewards = torch.stack([torch.tensor(reward_dict['reward']) for reward_dict in all_rewards])

    mean_reward = sum(reward['reward'] for reward in all_rewards) / len(all_rewards)
    mean_format_reward = sum(reward['format_reward'] for reward in all_rewards) / len(all_rewards)
    metadata = {
        'mean_reward': mean_reward,
        'mean_format_reward': mean_format_reward,
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal['mean', 'none'] = 'mean',
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal['std', 'none', 'mean'] = 'std',
):
    rewards = rearrange(raw_rewards, "(g n) -> g n", n = group_size)

    if baseline == 'mean':
        group_means = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - group_means
    elif baseline == 'none':
        advantages = rewards
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if advantage_normalizer == 'std':
        group_stds = rewards.std(dim=1, keepdim=True) + advantage_eps
        normalized_advantages = advantages / group_stds
    elif advantage_normalizer == 'mean':
        group_means = rewards.mean(dim=1, keepdim=True) + advantage_eps
        normalized_advantages = advantages / group_means
    elif advantage_normalizer == 'none':
        normalized_advantages = advantages
    else:
        raise ValueError(f"Unknown advantage normalizer: {advantage_normalizer}")

    metadata = {
        'mean': raw_rewards.mean().item(),
        'std': raw_rewards.std().item(),
        'min': raw_rewards.min().item(),
        'max': raw_rewards.max().item(),
    }
    normalized_advantages = rearrange(normalized_advantages, "g n -> (g n)", n = group_size)
    return normalized_advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo'] = 'none',
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    metadata = {}
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = rearrange(raw_rewards_or_advantages, "b -> b 1",)
    
    if importance_reweighting_method == 'none':
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
    else:
        raise NotImplementedError

    if response_mask is not None:
        per_token_policy_gradient_loss = per_token_policy_gradient_loss * response_mask

    return per_token_policy_gradient_loss, metadata
