import pandas as pd
import os
import argparse
from allennlp.predictors.predictor import Predictor
from extract_interpretations import evaluate_sentences


def get_tokens_dependencies_and_pos(sent, parser = 'allennlp'):
    '''
    Get PoS and dependency labels
    '''
    predictions = predictor.predict(sentence =  sent)
    return predictions['words'], predictions['pos'], predictions['predicted_dependencies']


def parse_sentences(data_dir, sentences_dir, parser = 'allennlp', ambig = True, data_type = 'NP-S',
                max_num_sentences = 100, parse_original = False, to_discard = None):
    '''
    Parse sentences
    If ambig = True: no cue and post-locus cue prompts; else pre-locus cues and pre&post-locus cue
    If parse_original: it derives the parse on the sentence pairs that the prompts are derived from
    '''

    output_dir = sentences_dir + '/parsed_' + parser + '/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data_file = data_dir + 'prompts_' + data_type + '_'
    if ambig:
        data_file += 'from_ambiguous'
    else:
        data_file += 'from_unambiguous'
    data_file += '.tsv'
    data_df = pd.read_csv(data_file, sep='\t', index_col='item')
    
    if to_discard != None:
        data_df = data_df[~data_df.index.isin(to_discard)]

    if parse_original:
        original_dir = sentences_dir.split(data_type)[0] + data_type + '/original_sentences/'
        if not os.path.isdir(original_dir):
            os.makedirs(original_dir)
        original_parses = original_dir + 'ambiguous' if ambig else original_dir + 'unambiguous'
        original_parses += '_parsed_' + parser + '.tsv'
        original_parses =  open(original_parses, 'w')
        original_parses.write('\t'.join(['id', 'tokens', 'pos', 'parse']) + '\n')

    for stimulus_id, row in data_df.iterrows():
        if parse_original:
            original_sent = row.sentence
            analyzed_sentence = get_tokens_dependencies_and_pos(original_sent, parser=parser)
            analyzed_sentence = [str(i) for i in list(analyzed_sentence)]
            original_parses.write(stimulus_id + '\t' + '\t'.join(analyzed_sentence) + '\n')
        prompt_type = 'nocue' if ambig else 'prelocuscue'
        prompt_pre_file = sentences_dir + stimulus_id + '_' + prompt_type + '.tsv'
        if os.path.exists(prompt_pre_file):
            with open(prompt_pre_file, 'r') as pre_file:
                with open(output_dir + stimulus_id + '_' + prompt_type + '_parsed.tsv', 'w') as pre_file_parsed:
                    pre_file_parsed.write('\t'.join(['id', 'tokens', 'pos', 'parse']) + '\n')
                    i = 0
                    for line in pre_file:
                        sent = line.replace('\n', '')
                        sent_id = stimulus_id + '_' + str(i + 1)
                        analyzed_sentence = get_tokens_dependencies_and_pos(sent, parser=parser)
                        analyzed_sentence = [str(i) for i in list(analyzed_sentence)]
                        pre_file_parsed.write(sent_id + '\t' + '\t'.join(analyzed_sentence) + '\n')
                        i += 1
        prompt_type = 'postlocuscue' if ambig else 'prepostlocuscue'
        prompt_post_file = sentences_dir + stimulus_id + '_' + prompt_type + '.tsv'
        if os.path.exists(prompt_post_file):
            with open(prompt_post_file) as post_file:
                with open(output_dir + stimulus_id + '_' + prompt_type + '_parsed.tsv', 'w') as post_file_parsed:
                    post_file_parsed.write('\t'.join(['id',  'tokens', 'pos', 'parse']) + '\n')
                    i = 0
                    for line in post_file:
                        sent = line.replace('\n', '')
                        sent_id = stimulus_id + '_' + str(i + 1)
                        analyzed_sentence = get_tokens_dependencies_and_pos(sent, parser=parser)
                        analyzed_sentence = [str(i) for i in list(analyzed_sentence)]
                        post_file_parsed.write(sent_id  + '\t'+ '\t'.join(analyzed_sentence) + '\n')
                        i += 1
    if parse_original: original_parses.close()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--parser', action="store", default='allennlp') # allennlp

    arg_parser.add_argument('--lm', action="store", default = 'LSTM') #LSTM, GPT2

    arg_parser.add_argument('--ambiguity', action="store", default='NP-S') # ambiguity type (NP-S, NP-Z, N-V_V, N-V_N)

    arg_parser.add_argument('--all', action="store_true", default=False) # Evaluate all generation setups within directory

    arg_parser.add_argument('--cuda', action="store_true", default=False) # Use GPU

    args = arg_parser.parse_args()


    generated_dir = 'generated/' + args.ambiguity + '/' + args.lm + '/'

    parser = args.parser
    
    #Load parser
    print('Loading parser', parser, '...')
    if parser == 'allennlp':
        predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
        if args.cuda:
            predictor._model = predictor._model.cuda()
    data_dir = 'data/'
    if not args.ambiguity == 'N-V':
        dirs_to_evaluate = [generated_dir + i + '/' for i in os.listdir(generated_dir) if not 'original_sentences' in i and not i.startswith('.')]
        i = 0
        for dir in dirs_to_evaluate:
            # If evaluating multiple setups, only run classification on original sentence pairs the first time
            parse_original = i == 0
            parse_sentences(data_dir, dir, parser=parser, ambig=True, data_type=args.ambiguity, parse_original = parse_original)
            parse_sentences(data_dir, dir, parser=parser, ambig=False, data_type=args.ambiguity,  parse_original = parse_original)
            i += 1
    else:
        to_discard = []
        for subtype in ['N-V_N', 'N-V_V']:
            df_ambig = pd.read_csv(data_dir + 'prompts_' + subtype + '_from_ambiguous.tsv', sep='\t')
            to_discard += list(df_ambig[df_ambig.disambiguation_type != 'agreement'].item)
            df_unambig = pd.read_csv(data_dir + 'prompts_' + subtype + '_from_unambiguous.tsv', sep='\t')
            to_discard += list(df_unambig[df_unambig.disambiguation_type != 'agreement'].item)
        to_discard = list(set(to_discard))
        for subtype in ['N-V_N', 'N-V_V']:
            generated_dir_tmp  = generated_dir.replace(args.ambiguity, args.ambiguity + '/' + subtype)
            dirs_to_evaluate = [generated_dir_tmp + i + '/' for i in os.listdir(generated_dir_tmp) if not 'original_sentences' in i and not i.startswith('.')]
            i = 0
            for dir in dirs_to_evaluate:
                # If evaluating multiple setups, only run classification on original sentence pairs the first time
                parse_original = i == 0
                parse_sentences(data_dir, dir, parser=parser, ambig=True, data_type=subtype, parse_original = parse_original, to_discard = to_discard)
                parse_sentences(data_dir, dir, parser=parser, ambig=False, data_type=subtype,  parse_original = parse_original, to_discard = to_discard)
                i += 1
