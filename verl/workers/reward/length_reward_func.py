import torch
import math
import numpy as np
import torch, math, numpy as np
from collections import defaultdict

from collections import defaultdict
# CosFn Reward
from typing import List

def normalize_entropy(entropy_values, method='log_minmax', clip=True):

    if not entropy_values:
        return []

    entropy = np.array(entropy_values, dtype=np.float32)

    if method == 'percentile':
        q_low = np.percentile(entropy, 5)
        q_high = np.percentile(entropy, 95)
        rng = max(1e-6, q_high - q_low)
        norm = (entropy - q_low) / rng

    elif method == 'log_minmax':
        log_entropy = np.log1p(entropy)  # log(1 + x)
        norm = (log_entropy - log_entropy.min()) / (log_entropy.max() - log_entropy.min() + 1e-8)

    elif method == 'zscore_sigmoid':
        mean = np.mean(entropy)
        std = np.std(entropy) + 1e-8
        z = (entropy - mean) / std
        norm = 1 / (1 + np.exp(-z))  # sigmoid

    elif method == 'minmax':
        norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    if clip:
        norm = np.clip(norm, 0.0, 1.0)

    return norm.tolist()




def sigmoid(x):
    return 1 / (1 + math.exp(-x))



def zipngram_tokens(tokens: List[int], ngram_size: int):
    return zip(*[tokens[i:] for i in range(ngram_size)])


def compute_repetition_penalty(
    tokens: List[int], gen_len: int, penalty: float, ngram_size: int, only_start: bool
) -> float:
    gen = tokens[:gen_len]
    repeated = []
    ngrams = set()

    for start_idx, ng in enumerate(zipngram_tokens(gen, ngram_size)):
        if ng in ngrams:
            repeated.append(start_idx)
        ngrams.add(ng)

    reward_vector = [0] * gen_len
    curr_end_idx = -1
    for start_idx in repeated:
        if not only_start or start_idx > curr_end_idx:
            for i in range(start_idx, min(start_idx + ngram_size, gen_len)):
                reward_vector[i] = penalty
        curr_end_idx = start_idx + ngram_size

    if gen_len == 0:
        return 0.0
    return sum(reward_vector) / gen_len  # average penalty over valid length


def entropy_length_reward_question_level(
    data, config, index, scores, id2complexity,
    response_length, reward_tensor, reward_metrics, meta
):
    entropy_values = data.batch['entropys']

    lambda_f = 0.5
    lambda_t = 0.5
    entropy_topk = config.entropy_topk
    alpha = config.alpha
    beta = config.beta
    norm_method = config.norm_method
    
    precomputed_entropy = []
    B = len(scores)
    for i in range(B):
        cur_entropy_all = entropy_values[i][:response_length[i]]
        top_k = max(1, int(len(cur_entropy_all) * entropy_topk))
        top_entropy = torch.topk(cur_entropy_all, top_k).values
        avg_entropy = top_entropy.mean().item() if top_k > 0 else 0.0
        precomputed_entropy.append(avg_entropy)
    if precomputed_entropy:
        normalized_entropy = normalize_entropy(precomputed_entropy, method=norm_method)
    else:
        normalized_entropy = []

    problem_stats = defaultdict(lambda: {
        'count': 0,
        'self_conf_correct_sum': 0.0,  
    })
    for i, score in enumerate(scores):
        gid = index[i]
        y = 1.0 if float(score['accuracy']) > 0 else 0.0
        e = normalized_entropy[i]             
        problem_stats[gid]['count'] += 1
        problem_stats[gid]['self_conf_correct_sum'] += y * (1.0 - e)*alpha +(1-alpha)*y

    id2difficulty = {}
    for gid, st in problem_stats.items():
        cnt = st['count']
        if cnt == 0:
            id2difficulty[gid] = 0.5
            continue
        scc = st['self_conf_correct_sum'] / cnt   
        difficulty = 1.0 - scc
        id2difficulty[gid] = difficulty

    all_difficulties = list(id2difficulty.values())
    
    cur_theta = np.quantile(all_difficulties, config.difficulty_threshold) if all_difficulties else 0.5
    theta_momentum = getattr(config, 'theta_momentum', 0.9)  
    if 'theta_difficulty' not in meta:
        meta['theta_difficulty'] = cur_theta
    else:
        meta['theta_difficulty'] = theta_momentum * meta['theta_difficulty'] + (1 - theta_momentum) * cur_theta

    theta_difficulty = meta['theta_difficulty']
    K = 2
    len_momentum = getattr(config, 'len_momentum', 0.9)
    if 'expect_len_bins' not in meta or len(meta['expect_len_bins']) != K:
        meta['expect_len_bins'] = [0.0 for _ in range(K)]
    expect_len_bins = meta['expect_len_bins']

    bin_len_sums   = [0.0 for _ in range(K)]
    bin_len_counts = [0   for _ in range(K)]
    bin_counts     = [0   for _ in range(K)]
    len_score_list = []
    total_response_len = []
    output_ids = data.batch['response_mask']   # token sequences
    repetition_penalty = -0.05               # fixed penalty per repeated token
    repetition_ngram_size = 40               # e.g., trigram
    repetition_only_start = True            # match original logic

    for i, score in enumerate(scores):
        gid = index[i]
        cur_difficulty = id2difficulty.get(gid, 0.5)
        cur_len        = int(response_length[i])
        cur_entropy    = normalized_entropy[i]
        confidence     = 1 - cur_entropy

        bin_id = 0 if cur_difficulty < theta_difficulty else 1

        bin_len_sums[bin_id]   += cur_len
        bin_len_counts[bin_id] += 1
        bin_counts[bin_id]     += 1
        total_response_len.append(cur_len)

        L_exp = expect_len_bins[bin_id] if expect_len_bins[bin_id] > 0 else cur_len
        L_exp += 1e-6

        len_score = 0.0
        margin = 0.2  

        if float(score['accuracy']) > 0 and bin_id == 0:
            penalty_strength = 0.5 + 0.5 * confidence
            len_score = max(0, 1 - cur_len / L_exp * penalty_strength) # 【0(flag),1】
            
        elif float(score['accuracy']) == 0 and bin_id == 1:
            length_boost = 0.5 + 0.5 * confidence
            len_score = min(1, cur_len / L_exp * length_boost-1) # 【，1（flag）】
            
        toks = output_ids[i]
        rep_penalty = compute_repetition_penalty(
            toks, cur_len,
            penalty=repetition_penalty,
            ngram_size=repetition_ngram_size,
            only_start=repetition_only_start
        )    
        len_score_total = len_score + rep_penalty
        
        len_score_list.append(len_score)

        total_reward = float(score['accuracy']) + lambda_f * float(score['format']) + lambda_t * len_score_total
        reward_tensor[i, cur_len - 1] = total_reward

        for key in score:
            reward_metrics[key].append(float(score[key]))
        reward_metrics['overall_w_len'].append(total_reward)
        reward_metrics['repetition_penalty'].append(rep_penalty)
        reward_metrics['difficulty'].append(cur_difficulty)
        reward_metrics['1-difficulty'].append(1 - cur_difficulty)
        reward_metrics['confidence'].append(confidence)
        reward_metrics['raw_entropy'].append(precomputed_entropy[i])
        reward_metrics['len_score'].append(len_score)
        reward_metrics['response_length'].append(cur_len)
        reward_metrics['difficulty_bin_id'].append(bin_id)

    for j in range(K):
        if bin_len_counts[j] > 0:
            avg_len = bin_len_sums[j] / bin_len_counts[j]
            if expect_len_bins[j] == 0.0:
                expect_len_bins[j] = avg_len
            else:
                expect_len_bins[j] = len_momentum * expect_len_bins[j] + (1 - len_momentum) * avg_len
    reward_metrics['len_momentum_curve'].append(len_momentum)
    reward_metrics['L_avg'] = [np.mean(total_response_len) if total_response_len else 0.0]
    for j in range(K):
        reward_metrics[f'L_exp_bin_{j}'] = [expect_len_bins[j]]
        reward_metrics[f'bin_count_{j}'] = [bin_counts[j]]

    reward_metrics['max_difficulty'] = [max(all_difficulties) if all_difficulties else 0.0]
    reward_metrics['min_difficulty'] = [min(all_difficulties) if all_difficulties else 0.0]
    reward_metrics['mean_len_score'] = [np.mean(len_score_list) if len_score_list else 0.0]
    reward_metrics['max_len_score']  = [max(len_score_list) if len_score_list else 0.0]
    reward_metrics['theta_batch']  = [cur_theta]
    reward_metrics['theta_EMA']  = [theta_difficulty]

    return reward_tensor, reward_metrics, meta
