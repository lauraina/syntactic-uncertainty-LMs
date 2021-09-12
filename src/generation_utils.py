
# Based on and modified version of code in
# https://github.com/huggingface/transformers/blob/9d94aecd516c7540a94b9d781ef28d7375a796bc/src/transformers/generation_utils.py

# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import os
import torch
from torch import nn
from torch.nn import functional as F



def prepare_inputs_for_generation(input_ids, **kwargs):
    return {"input_ids": input_ids}

def do_output_past(lm, outputs):
    """During generation, decide whether to pass the `past` variable to the next forward pass."""
    has_output_past = getattr(lm.config, "output_past", False)
    mem_len = getattr(lm.config, "mem_len", 0)
    if len(outputs) <= 1:
        return False
    if mem_len > 0 or has_output_past:
        return True
    return False

def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def generate(
    input_ids=None,
    lm = None,
    lm_vocabulary = None,
    max_length=20,
    do_sample=False,
    num_beams=1,
    temperature=1,
    top_k=5,
    top_p=1,
    repetition_penalty=1,
    bos_token_id='<|endoftext|>',
    pad_token_id=None,
    eos_token_ids=None,
    length_penalty=1,
    num_return_sequences=1,
    unknown_penalty = 1,
    unk_token_id = None):
    # We cannot generate if the model does not have a LM head

    if lm_vocabulary.transformer:
        if lm.get_output_embeddings() is None:
            # raise AttributeError(
            #     "You tried to generate sequences with a model that does not have a LM Head."
            #     "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            # )
            pass
        eos_token_ids = lm_vocabulary.tokenizer.eos_token_id
        bos_token_id = None if lm_vocabulary.tokenizer._bos_token == None else lm_vocabulary.tokenizer.bos_token_id
    else:
        # MODIFIED: adapted to LSTM
        eos_token_ids = lm_vocabulary.encode('<eos>')
        unk_token_id = lm_vocabulary.encode('<unk>')

    #pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    # assert pad_token_id is None or (
    #     isinstance(pad_token_id, int) and (pad_token_id >= 0)
    # ), "`pad_token_id` should be a positive integer."
    assert (eos_token_ids is None) or (
        isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
    ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(lm.parameters()).device
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # current position and vocab size
    cur_len = input_ids.shape[1]
    vocab_size = lm_vocabulary.size

    if num_return_sequences != 1 and do_sample:
        # Expand input to num return sequences
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
        input_ids = input_ids.contiguous().view(
            batch_size * num_return_sequences, cur_len
        )  # shape: (batch_size * num_return_sequences, cur_len)
        effective_batch_size = batch_size * num_return_sequences
    else:
        effective_batch_size = batch_size

    if num_beams > 1:
        output = generate_beam_search(
            input_ids, lm, lm_vocabulary,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            unk_token_id,
            effective_batch_size,
            num_return_sequences,
            length_penalty,
            unknown_penalty,
            num_beams,
            vocab_size,
            )
    else:
        output = generate_no_beam_search(
            input_ids, lm, lm_vocabulary,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            unk_token_id,
            unknown_penalty,
            effective_batch_size,
        )


    return output


def generate_no_beam_search(
    input_ids ,lm, lm_vocabulary,
    cur_len,
    max_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    pad_token_id,
    eos_token_ids,
    unk_token_id,
    unknown_penalty,
    batch_size = 1,):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)
    i = 0
    past = None if lm_vocabulary.transformer else lm.init_hidden(batch_size) #MODIFIED: adapt to LSTM
    while cur_len < max_length:

        model_inputs = prepare_inputs_for_generation(input_ids, past=past)
        
        if lm_vocabulary.transformer:
            outputs = lm(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            if do_output_past(lm, outputs):
                # if model has past, then set the past variable to speed up decoding
                past = outputs[1]
        else: #MODIFIED: adapt to LSTM
            outputs = lm(input_ids.t(), past)
            past = outputs[1]
            # next_token_logits = outputs[0].view(-1, lm_vocabulary.size)[-1]
            next_token_logits = outputs[0][-1]
        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)
        # if i == 0:
        #     enforce_start_of_continuation_(next_token_logits, batch_size, 1, input_ids, 100000000000000000)

        if unknown_penalty != 1.0 and unk_token_id != None:
            for i in range(batch_size):
                if next_token_logits[i, unk_token_id]  < 0:
                    next_token_logits[i, unk_token_id] *= unknown_penalty
                else:
                    next_token_logits[i, unk_token_id] /= unknown_penalty

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p/top-k filtering
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        tokens_to_add = next_token

        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        if eos_token_ids is not None:
            for eos_token_id in eos_token_ids:
                eos_in_sents = tokens_to_add == eos_token_id # finished sentence
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

        cur_len = cur_len + 1
        i += 1

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

    # # if there are different sentences lengths in the batch, some batches have to be padded
    # if sent_lengths.min().item() != sent_lengths.max().item():
    #     assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
    #     # finished sents are filled with pad_token
    #     decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)

    decoded = input_ids

    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    return decoded

def generate_beam_search(
    input_ids, lm, lm_vocabulary,
    cur_len,
    max_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    pad_token_id,
    eos_token_ids,
    unk_token_id,
    batch_size,
    num_return_sequences,
    length_penalty,
    unknown_penalty,
    num_beams,
    vocab_size,
):
    """ Generate sequences for each example with beam search.
    """

    # Expand input to num beams
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
    input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # Greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)


    # cache compute states
    past = None if lm_vocabulary.transformer else lm.init_hidden(num_beams) #MODIFIED: adapt to LSTM

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        torch.cuda.empty_cache()
        model_inputs = prepare_inputs_for_generation(input_ids, past=past)

        if lm_vocabulary.transformer:
            torch.cuda.empty_cache()
            outputs = lm(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            if do_output_past(lm, outputs):
                # if model has past, then set the past variable to speed up decoding
                past = outputs[1]
        else: #MODIFIED: adapt to LSTM
            outputs = lm(input_ids.T, past)
            past = outputs[1]
            # next_token_logits = outputs[0].view(-1, lm_vocabulary.size)[-1]
            next_token_logits = outputs[0][-1]

        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)


        if unknown_penalty != 1.0 and unk_token_id != None:

            for i in range(batch_size):
                if next_token_logits[i, unk_token_id]  < 0:
                    next_token_logits[i, unk_token_id] *= unknown_penalty
                else:
                    next_token_logits[i, unk_token_id] /= unknown_penalty
        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            next_tokens = torch.multinomial(
                F.softmax(_scores, dim=-1), num_samples=2 * num_beams
            )  # (batch_size, num_beams * 2)

            # Compute next scores
            next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)

        else:
            torch.cuda.empty_cache()
            # do greedy beam search
            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            assert scores.size() == (batch_size * num_beams, vocab_size)
            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence

        for batch_idx in range(batch_size):
            torch.cuda.empty_cache()
            # if we are done with this sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for idx, score in zip(next_tokens[batch_idx], next_scores[batch_idx]):

                # get beam and word IDs
                beam_id = idx // vocab_size
                token_id = idx % vocab_size

                # # add to generated hypotheses if end of sentence or last iteration
                # if eos_token_ids is not None and token_id.item() in eos_token_ids:
                #     generated_hyps[batch_idx].add(
                #         input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item(),
                #     )
                # else:
                #     # add next predicted word if it is not eos_token
                next_sent_beam.append((score, token_id, batch_idx * num_beams + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break
            torch.cuda.empty_cache()

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

        # re-order internal states
        if past:
            past = _reorder_cache(past, beam_idx)

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    for batch_idx in range(batch_size):
        # Add all open beam hypothesis to generated_hyps
        if not done[batch_idx]:
            for idx, score in zip(next_tokens[batch_idx], next_scores[batch_idx]):

                # get beam and word IDs
                beam_id = idx // vocab_size
                token_id = idx % vocab_size
                generated_hyps[batch_idx].add(
                    input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                )

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    decoded = best
    # shorter batches are filled with pad_token


    #assert pad_token_id is not None, "`Pad_token_id` has to be defined"
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_ids[0]
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(lm.parameters()).device)

    return decoded

def _reorder_cache(past, beam_idx):
    reordered_past = []
    for layer_past in past:
        # get the correct batch idx from layer past batch dim
        # batch dim of `past` and `mems` is at 2nd position
        reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
        reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
        # check that shape matches
        assert reordered_layer_past.shape == layer_past.shape
        reordered_past.append(reordered_layer_past)
    past = tuple(reordered_past)
    return past



def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

#
class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
            
# ADDED: code to extract token probabilities of completions

def get_probability_completion(
    prompt_ids=None,
    completion_ids = None,
    lm = None,
    lm_vocabulary = None):
    '''
    Return set of log probabilities for each token in completion_ids
    '''

    if lm_vocabulary.transformer:
        if lm.get_output_embeddings() is None:
            # raise AttributeError(
            #     "You tried to generate sequences with a model that does not have a LM Head."
            #     "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            # )
            pass
        eos_token_ids = lm_vocabulary.tokenizer.eos_token_id
        bos_token_id = None if lm_vocabulary.tokenizer._bos_token == None else lm_vocabulary.tokenizer.bos_token_id
    else:
        eos_token_ids = lm_vocabulary.encode('<eos>')
        unk_token_id = lm_vocabulary.encode('<unk>')

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    batch_size = 1

    # current position and vocab size
    cur_len = prompt_ids.shape[1]
    
    len_prompt = prompt_ids.shape[1]
    len_completion = len(completion_ids)
    len_sentence = len_prompt + len_completion
    vocab_size = lm_vocabulary.size
    
    i = 0
    past = None if lm_vocabulary.transformer else lm.init_hidden(batch_size)
    
    probs = torch.Tensor([])
    
    while cur_len < len_sentence:

        model_inputs = prepare_inputs_for_generation(prompt_ids, past=past)

        if lm_vocabulary.transformer:
            outputs = lm(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            if do_output_past(lm, outputs):
                # if model has past, then set the past variable to speed up decoding
                past = outputs[1]
        else:
            outputs = lm(prompt_ids.t(), past)
            past = outputs[1]
            next_token_logits = outputs[0][-1]
        
        next_token_probabilities = F.log_softmax(next_token_logits, dim=-1).cpu().detach()
        next_token = completion_ids[i]
        next_token_prob = next_token_probabilities[0][next_token]
        
        probs = torch.cat([probs, next_token_prob.unsqueeze(-1)], dim=-1)

        tokens_to_add = next_token.unsqueeze(-1)
    
        prompt_ids = torch.cat([prompt_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        
        cur_len = cur_len + 1
        i += 1

    return probs
