from lm_utils import Vocabulary, load_lm
from generation_utils import *
import pandas as pd
import torch
import os
from transformers import *
import argparse
from itertools import product
from spacy.lang.en import English
from spacy.pipeline import Sentencizer

# Load Spacy tools
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
#nlp.add_pipe("sentencizer")

def get_sentences(prompt_text, lm, lm_vocabulary, repetition_penalty = 1, num_return_sequences = 1, temperature = 1,
                  unknown_penalty = 10000000000000000, top_k = 50, num_beams = 1, max_length = 30, do_sample = False, top_p = 1,
                  generation_type = 'greedy', cuda = False):
    '''
    Generate set of completions from prompt
    '''
    
    prompt = lm_vocabulary.encode(prompt_text)

    if not lm_vocabulary.type in 'LSTM':
        # If vocabulary with subwords, generate more tokens
        max_length = 50
    max_length = len(prompt) + max_length

    sentences_generated = []
    prompt = torch.LongTensor(prompt).unsqueeze(0)
    if cuda:
        prompt = prompt.cuda()
    if generation_type == 'beam-search':
        num_return_sequences = num_beams
    
    generated = generate(prompt, lm, lm_vocabulary, do_sample=do_sample,
                         repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, temperature=temperature,
                         unknown_penalty=unknown_penalty, top_k=top_k, top_p = top_p, num_beams=num_beams, max_length=max_length)  # generate sequence

    for i in range(len(generated)):
        chunk = lm_vocabulary.decode(generated[i], skip_special_tokens=True)
        # Crop to sentence boundaries
        chunk = chunk.replace('\n', ' ').replace('\t', ' ').rstrip()
        sent = list(nlp(chunk).sents)[0].text.rstrip()
        sentences_generated.append(sent)
        if generation_type == 'beam-search' and i == 0:
            break
    return sentences_generated


def get_hyperparameters_combinations(generation_type, num_return_sequences = 100):
    '''
    Generate hyperparameter combinations for different decoding strategies
    '''
    do_sample = [False]
    num_beams = [1]
    temperature = [0.6, 0.75, 0.9, 1]
    top_p = [1]
    top_k = [0]
    num_return_sequences = [num_return_sequences] # constant
    repetition_penalty = [1] # constant

    if generation_type == 'beam-search':
        num_beams = [16]
        temperature = [1]
    elif generation_type == 'sampling-k':
        top_k = [80, 160, 320, 640]
        do_sample = [True]
    elif generation_type == 'sampling-p':
        top_p = [0.6, 0.75, 0.90, 1]
        do_sample = [True]

    generation_type = [generation_type]
    hyperparameters = [generation_type, do_sample, num_beams, temperature, top_p, top_k, num_return_sequences, repetition_penalty]
    hyperparameters_combinations = list(product(*hyperparameters))
    return hyperparameters_combinations


def get_tokenized_sentence(sent):
    return ' '.join([str(t) for t in nlp(sent)])

def get_setup_name(generation_type, do_sample, num_beams, temperature, top_p, top_k, num_return_sequences, repetition_penalty):
    '''
    From generation hyperparameters to setup name
    '''
    parameters_id = generation_type
    if do_sample:
        if top_p < 1 or (top_p == 1 and top_k == 0):
            parameters_id += '_p' + str(top_p)
        elif top_k != 0:
            parameters_id += '_k' + str(top_k)

    if generation_type == 'beam-search':
        parameters_id += '_beams' + str(num_beams)

    parameters_id += '_temperature' + str(temperature)

    parameters_id += '_repetition' + str(repetition_penalty)
    return parameters_id


def check_if_unknown(prompt):
    decoded_prompt = lm_vocabulary.decode(lm_vocabulary.encode(prompt))
    return  '<unk>' in decoded_prompt


def generate_sentences(data_dir, lm, lm_vocabulary, output_dir, generation_type, ambig = True, num_return_sequences = 100, hyperparameters = None,
                       repetition_penalty = 1,  max_length= 30, cuda = False, category = None, data_type = 'NP-S', datapoints = "all", to_discard = None):
    '''
    Generate completions from prompts
    If ambig = True: no cue and post-locus cue prompts; else pre-locus cues and pre&post-locus cue
    '''
    if hyperparameters == None:
        # If generation using multiple hyperparameters combinations: get list of setups
        hyperparameters_combinations = get_hyperparameters_combinations(generation_type, num_return_sequences = num_return_sequences)
    else:
        hyperparameters_combinations = [hyperparameters]

    for hyperparameters in hyperparameters_combinations:
    
        generation_type, do_sample, num_beams, temperature, top_p, top_k, num_return_sequences, repetition_penalty = hyperparameters
        parameters_id = get_setup_name(generation_type, do_sample, num_beams, temperature, top_p, top_k, num_return_sequences, repetition_penalty)
        output_dir_tmp = output_dir + parameters_id + '/'

#        if ambig:
#            output_dir_tmp += 'from_ambiguous/'
#        else:
#            output_dir_tmp += 'from_unambiguous/'

        if not os.path.isdir(output_dir_tmp):
            os.makedirs(output_dir_tmp)

        # If using LSTM, the input needs to be tokenized
        tokenize = lm_vocabulary.type == 'LSTM'
        
        #Load prompts data from file
        data_file = data_dir + 'prompts_' + data_type + '_'
        if ambig:
            data_file += 'from_ambiguous'
        else:
            data_file += 'from_unambiguous'
        data_file += '.tsv'
        data_df = pd.read_csv(data_file, sep='\t', index_col='item')
        
        if to_discard != None:
            data_df = data_df[~data_df.index.isin(to_discard)]

        if datapoints != 'all':
            # Filter data if a list of datapoints is given
            datapoints = eval(datapoints)
            data_df = data_df[data_df.index.isin(datapoints)]

        for index, row in data_df.iterrows():
            index = index.replace('/', '-')
            
            # Prompt_pre : no cue prompt if ambig else pre-locus cue prompt
            # Prompt_post: post-locus cue prompt if ambig else pre&postlocus cue prompt
            prompt_pre, prompt_post = row.prompt_pre, row.prompt_post
            if tokenize:
                prompt_pre, prompt_post = get_tokenized_sentence(prompt_pre), get_tokenized_sentence(prompt_post)
            
            print('Generating from:', prompt_pre)
            # Generate sentences from prompts
            sentences_pre = get_sentences(prompt_pre, lm, lm_vocabulary, num_beams= num_beams,
                                              do_sample = do_sample, top_p = top_p, top_k = top_k, temperature = temperature,
                                              num_return_sequences=num_return_sequences, generation_type = generation_type,
                                              repetition_penalty=repetition_penalty, max_length = max_length, cuda = cuda)
                            
            print('Generating from:', prompt_post)
            sentences_post = get_sentences(prompt_post, lm, lm_vocabulary, num_beams= num_beams,
                                            do_sample = do_sample, top_p = top_p, top_k = top_k, temperature = temperature,
                                            num_return_sequences=num_return_sequences, generation_type = generation_type,
                                            repetition_penalty=repetition_penalty, max_length = max_length, cuda = cuda)
                                            
            # Write sentences to files
            prompt_type = 'nocue' if ambig else 'prelocuscue'
            with open(output_dir_tmp + index + '_'+ prompt_type + '.tsv', 'w') as output_file_pre:
                for i in range(len(sentences_pre)):
                    output_file_pre.write(sentences_pre[i] +  '\n')
            prompt_type = 'postlocuscue' if ambig else 'prepostlocuscue'
            with open(output_dir_tmp + index + '_' + prompt_type +'.tsv', 'w') as output_file_post:
                for i in range(len(sentences_post)):
                    output_file_post.write(sentences_post[i]  + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lm', action="store", default='LSTM') # LM to evaluate (GPT2, LSTM)

    parser.add_argument('--ambiguity', action="store", default='NP-S') # Ambiguity type: NP-S, NP-Z or N-V

    parser.add_argument('--datapoints', action="store", default='all') # List of specific items (ids) to consider
    
    # Decoding strategy: sampling-p (nucleus sampling, by default with p = 1, pure sampling), sampling-k (top-k sampling; if k=1, greedy decoding),  beam-search
    parser.add_argument('--generation_type', action="store", default='sampling-p')

    parser.add_argument('--num_completions', action = "store", default = 100, type = int) # Number of completions to generate

    parser.add_argument('--search', action="store_true",  default=False) # Consider differerent generation hyperparameters

    parser.add_argument('--num_beams', action="store", default=16, type = int) # Beam size for beam search

    parser.add_argument('--top_p', action = 'store', default = 1, type = float) # Nucleus size for sampling

    parser.add_argument('--top_k', action='store', default=0, type = int) # Top k for top-k sampling

    parser.add_argument('--all', action='store_true', default=False) # Consider predefined set of decoding strategies; else, default one

    parser.add_argument('--temperature', action='store', default=1, type = float) # Temperature parameter

    parser.add_argument('--max_length', action='store', default=30, type = int) # Fixed number of tokens to generate

    parser.add_argument('--repetition_penalty', action='store', default=1, type = float) # Repetition penalty

    parser.add_argument('--cuda', action='store_true', default=False) # Use GPU

    args = parser.parse_args()

    print('Loading language model', args.lm, '...' )
    lm, lm_vocabulary = load_lm(model_type = args.lm, cuda = args.cuda) # Load language model and vocabulary

    if not args.all:
        generation_types = [args.generation_type]
    else:
        generation_types = ['sampling-p', 'beam-search']

    data_dir = 'data/'

    for generation_type in generation_types:
        if args.search:
            hyperparameters = None
        else:
            if generation_type.startswith('sampling'):
                args.num_beams = 1
                sample = True
                if args.generation_type == 'sampling-p':
                    args.top_k = 0
                if args.generation_type == 'sampling-k':
                    args.top_p = 1
            else:
                sample = False
                args.num_beams = 16
            hyperparameters = [generation_type, sample, args.num_beams, args.temperature, args.top_p, args.top_k, args.num_completions, args.repetition_penalty]
        
        if args.ambiguity != 'N-V':
            output_dir = 'generated/' + args.ambiguity.replace('/', '-') + '/' + args.lm + '/'
            # ambig = True: generating from prompts with temporary ambiguity (No cue, pre-locus cue)
            generate_sentences(data_dir, lm, lm_vocabulary, output_dir,generation_type, ambig = True, hyperparameters = hyperparameters,
                               repetition_penalty = args.repetition_penalty, num_return_sequences=args.num_completions, max_length = args.max_length,
                               cuda = args.cuda, data_type=args.ambiguity, datapoints = args.datapoints)
            
            # ambig = True: generating from prompts without temporary ambiguity (pre-locus cue, pre&post-locus cues)
            generate_sentences(data_dir, lm, lm_vocabulary, output_dir,generation_type, ambig = False,hyperparameters = hyperparameters,
                               repetition_penalty=args.repetition_penalty, num_return_sequences=args.num_completions, max_length = args.max_length,
                               cuda = args.cuda, data_type=args.ambiguity, datapoints = args.datapoints)
        else:
            to_discard = []
            for subtype in ['N-V_N', 'N-V_V']:
                df_ambig = pd.read_csv(data_dir + 'prompts_' + subtype + '_from_ambiguous.tsv', sep='\t')
                to_discard += list(df_ambig[df_ambig.disambiguation_type != 'agreement'].item)
                df_unambig = pd.read_csv(data_dir + 'prompts_' + subtype + '_from_unambiguous.tsv', sep='\t')
                to_discard += list(df_unambig[df_unambig.disambiguation_type != 'agreement'].item)
            to_discard = list(set(to_discard))
    
            for subtype in ['N-V_N', 'N-V_V']:
                output_dir = 'generated/N-V/' + subtype + '/' + args.lm + '/'
                # ambig = True: generating from prompts with temporary ambiguity (No cue, pre-locus cue)
                generate_sentences(data_dir, lm, lm_vocabulary, output_dir,generation_type, ambig = True, hyperparameters = hyperparameters,
                               repetition_penalty = args.repetition_penalty, num_return_sequences=args.num_completions, max_length = args.max_length,
                               cuda = args.cuda, data_type=subtype, datapoints = args.datapoints, to_discard = to_discard)
            
                # ambig = True: generating from prompts without temporary ambiguity (pre-locus cue, pre&post-locus cues)
                generate_sentences(data_dir, lm, lm_vocabulary, output_dir,generation_type, ambig = False,hyperparameters = hyperparameters,
                               repetition_penalty=args.repetition_penalty, num_return_sequences=args.num_completions, max_length = args.max_length,
                               cuda = args.cuda, data_type=subtype, datapoints = args.datapoints, to_discard = to_discard)
    
