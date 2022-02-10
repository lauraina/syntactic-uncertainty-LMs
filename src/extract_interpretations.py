import pandas as pd
import os
import argparse

def nsubj_or_dobj(parse, tokens, simple = False):
    '''
    Subject vs. direct object rules
    '''
    if simple:
        if parse[0].startswith('nsubj'):
            return 'nsubj'
        elif parse[0] == 'dobj': # correcting mislabeled S cases
            if parse[1] in ['root', 'auxpass', 'aux', 'ccomp', 'cop', 'parataxis']:
                return 'nsubj'
            else:
                return 'dobj'
        else:
            return 'other'
    else:
        if parse[0].startswith('nsubj'):
            return 'nsubj'
        elif parse[0] == 'dobj': # correcting mislabeled S cases
            if parse[1] in ['root', 'auxpass', 'aux', 'cop', 'ccomp', 'parataxis'] and tokens[1] != 'to':
                return 'nsubj'
            else:
                return 'dobj'
            return 'dobj'
        elif parse[0] == 'cc':
            if len(parse)> 2 and parse[1] == 'conj':
                    return nsubj_or_dobj(parse[2:], tokens[2:])
            else:
                return 'other'
        elif parse[0] in ['nn', 'poss', 'amod']: # part of a complex NP
            if len(parse) > 1:
                if parse[1] == 'possessive' or parse[1] == 'punct':
                    if parse[1] == 'possessive' and parse[2] == 'punct':
                        if len(parse) > 3:
                            return nsubj_or_dobj(parse[3:],  tokens[3:])
                        else:
                            return 'other'
                    if len(parse) > 2:
                        return nsubj_or_dobj(parse[2:], tokens[2:])
                    else:
                        return 'other'
                else:
                    return nsubj_or_dobj(parse[1:], tokens[1:])
            else:
                return 'other'
        else:
            return 'other'
        


def get_interpretation(tokens, parse, pos, locus_ambiguity, data ='NP-S', with_postlocus_cue = False, postlocus_cue = None, occurrence_locus_ambiguity = 1, occurrence_postlocus_cue  = 1, parser = 'allennlp'):
    '''
    Classify completion based on interpretation of locus of ambiguity
    '''
    parse = [p.lower() for p in parse]
    
    indices_locus_ambiguity = [i for i, val in enumerate(tokens) if val == locus_ambiguity] # Find locus of ambiguity in case there are multiple instances of this word type
    index_ambig_word = indices_locus_ambiguity[occurrence_locus_ambiguity -1 ]
    if postlocus_cue != None: # Find disambiguator (e.g. post-locus cue) if there are multiple instances of this word type
        indices_postlocus_cue = [i for i, val in enumerate(tokens) if val == postlocus_cue]
        index_postlocus_cue = indices_postlocus_cue[occurrence_postlocus_cue - 1]

    if data == 'NP-S':
        parse_type_convert = {'nsubj': 'S', 'dobj': 'NP', 'other': 'other'}
        if not with_postlocus_cue:
            parse_type = nsubj_or_dobj(parse[index_ambig_word:], tokens[index_ambig_word:])
        else:
            parse_type = nsubj_or_dobj(parse[index_ambig_word:], tokens[index_ambig_word:], simple = True)
        parse_type = parse_type_convert[parse_type]
        detailed = parse_type
    elif data == 'NP-Z':
        parse_type_convert = {'nsubj': 'Z', 'dobj': 'NP', 'other': 'other'}
        if not with_postlocus_cue:
            parse_type = nsubj_or_dobj(parse[index_ambig_word:],tokens[index_ambig_word:])
        else:
            parse_type = nsubj_or_dobj(parse[index_ambig_word:], tokens[index_ambig_word:], simple = True)
        parse_type = parse_type_convert[parse_type]
        detailed = parse_type
        # If after post-locus cue: check if this is embedded in the subordinate close (not root, and with comma - not followed by conjunction - before root)
        if with_postlocus_cue:
            if parse[index_ambig_word +1] in ['cop', 'ccomp', 'parataxis']:
                indices_comma = [i for i, val in enumerate(tokens) if val == ',']
                for idx_comma in indices_comma:
                    if idx_comma > index_ambig_word:
                        if 'root' in parse[idx_comma:]:
                            if not parse[indices_comma[0] +1] == 'cc':
                                detailed += '_mainverb_in_subordinate'
                                break
                        else:
                            break
    elif data.startswith('N-V'):
        parse_type = pos[index_ambig_word]
        detailed = parse_type + '_' + parse[index_ambig_word]
        if parse_type == 'MD': parse_type = 'V'
        if parse_type[0] not in ['N', 'V']:
            parse_type = 'other'
        else:
            parse_type = parse_type[0]
    return parse_type, detailed

def evaluate_original_sentences(data_dir, data_type ='NP-S', parser ='allennlp', ambig = True):
    '''
    Extract interpretations from generated sentences
    '''
    data_file = data_dir + 'prompts_' + data_type + '_'
    if ambig:
        data_file += 'from_ambiguous'
    else:
        data_file += 'from_unambiguous'
    data_file += '.tsv'
    data_df = pd.read_csv(data_file, sep='\t', index_col='item')

    parse_dir = generated_dir + '/parsed_' + parser + '/'

    output_dir = generated_dir + 'interpretations_' + parser + '/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not data_type.startswith('N-V'):
        correct_parse = data_type.split('-')[-1]
    else:
        correct_parse = data_type.split('_')[-1]
    original_dir = 'generated/' + data_type + '/original_sentences/' if not data_type.startswith('N-V') else 'generated/N-V/' + data_type + '/original_sentences/'
    if not os.path.isdir(original_dir):
        os.makedirs(original_dir)
    original_parses = original_dir + 'ambiguous' if ambig else  original_dir + 'unambiguous'
    original_parses += '_parsed_' + parser + '.tsv'
    original_df = pd.read_csv(original_parses, sep = '\t', index_col = "id")
    original_output = original_dir + 'ambiguous' if ambig else  original_dir + 'unambiguous'
    original_output = open(original_output + '_interpretations.tsv', 'w')
    original_output.write('\t'.join(['item_id', 'sentence', 'interpretation', 'detailed']) + '\n')

    sentences_original ={item: sent for item, sent in zip(data_df.index, data_df.sentence)}
    
    incorrect = []
    for item_id, row in data_df.iterrows():
        parsed_row = original_df[original_df.index == item_id]
        if len(parsed_row) == 1:
            parse = eval(parsed_row.parse[0])
            tokens = eval(parsed_row.tokens[0])
            pos = eval(parsed_row.pos[0])
            locus_ambiguity = row.locus_ambiguity
            postlocus_cue = row.postlocus_cue
            if postlocus_cue == "hadn't": postlocus_cue = 'had'
            interpretation, detailed = get_interpretation(tokens, parse, pos,  locus_ambiguity, data=data_type, postlocus_cue = postlocus_cue, with_postlocus_cue= True, parser = parser)
            sentence = sentences_original[item_id]
            if interpretation != correct_parse:
                incorrect.append(item_id)
            original_output.write('\t'.join([item_id, sentence, interpretation, detailed]) + '\n')
    original_output.close()
    return incorrect
    
def evaluate_sentences(data_dir, generated_dir, data_type ='NP-S', parser ='allennlp', ambig = True, parse_original = False, to_discard = []):
    '''
    Extract interpretations from generated sentences
    '''
    data_file = data_dir + 'prompts_' + data_type + '_'
    if ambig:
        data_file += 'from_ambiguous'
    else:
        data_file += 'from_unambiguous'
    data_file += '.tsv'
    data_df = pd.read_csv(data_file, sep='\t', index_col='item')

    parse_dir = generated_dir + '/parsed_' + parser + '/'

    output_dir = generated_dir + 'interpretations_' + parser + '/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    prompt_type_pre = 'nocue' if ambig else 'prelocuscue'
    output_file_pre = open(output_dir + 'interpretations_'+ prompt_type_pre + '.tsv', 'w')
    output_file_pre.write('\t'.join(['item_id', 'sent_id', 'sentence', 'interpretation', 'detailed']) + '\n')
    prompt_type_post = 'postlocuscue' if ambig else 'prepostlocuscue'
    output_file_post = open(output_dir + 'interpretations_' + prompt_type_post + '.tsv', 'w')
    output_file_post.write('\t'.join(['item_id', 'sent_id', 'sentence', 'interpretation', 'detailed']) + '\n')

    for item_id, row in data_df.iterrows():
        item_id = item_id.replace('/', '-')
        if not item_id in to_discard:
            if os.path.exists(parse_dir + item_id + '_' + prompt_type_pre + '_parsed.tsv') and os.path.exists(parse_dir + item_id + '_'+ prompt_type_post +'_parsed.tsv'):
                prompt_pre_data = pd.read_csv(parse_dir + item_id + '_' + prompt_type_pre +'_parsed.tsv',sep = '\t', index_col = "id")[:100]
                prompt_post_data = pd.read_csv(parse_dir + item_id + '_'+ prompt_type_post +'_parsed.tsv', sep = '\t', index_col = "id")[:100]
                generated_pre = open(generated_dir + item_id + '_' + prompt_type_pre +'.tsv', 'r').readlines()
                generated_pre = [i.replace('\n', '') for i in generated_pre]
                generated_post = open(generated_dir + item_id + '_' + prompt_type_post +'.tsv', 'r').readlines()
                generated_post = [i.replace('\n', '') for i in generated_post]
                sent_number = 0
                for sent_id, row_pre in prompt_pre_data.iterrows():
                    parse = eval(row_pre.parse)
                    tokens = eval(row_pre.tokens)
                    pos = eval(row_pre.pos)
                    locus_ambiguity, postlocus_cue = row.locus_ambiguity, row.postlocus_cue
                    occurrence_locus_ambiguity, occurrence_postlocus_cue = row.occurrence_locus_ambiguity, row.occurrence_postlocus_cue
                    sentence = generated_pre[sent_number]
                    indices_locus_ambiguity = [i for i, val in enumerate(tokens) if val == locus_ambiguity]
                    locus_ambiguity_as_subword = False # The locus of ambiguity is treated as a subword
                    for t in tokens:
                        if t.startswith(locus_ambiguity):
                            if not t.endswith(locus_ambiguity):
                                locus_ambiguity_as_subword = True
                            break
                    if (locus_ambiguity in tokens and len(indices_locus_ambiguity) >= occurrence_locus_ambiguity) and not locus_ambiguity_as_subword:
                        interpretation, detailed = get_interpretation(tokens, parse, pos, locus_ambiguity, data = data_type, parser = parser)
                        output_file_pre.write('\t'.join([item_id, sent_id,  sentence, interpretation, detailed]) + '\n')
                    else:
                        pass
                    sent_number += 1
                sent_number = 0
                for sent_id, row_post in prompt_post_data.iterrows():
                    parse = eval(row_post.parse)
                    tokens = eval(row_post.tokens)
                    pos = eval(row_post.pos)
                    locus_ambiguity, postlocus_cue = row.locus_ambiguity, row.postlocus_cue
                    occurrence_locus_ambiguity, occurrence_postlocus_cue = row.occurrence_locus_ambiguity, row.occurrence_postlocus_cue
                    if postlocus_cue == "hadn't": postlocus_cue = 'had'
                    indices_locus_ambiguity = [i for i, val in enumerate(tokens) if val == locus_ambiguity]
                    indices_postlocus_cue = [i for i, val in enumerate(tokens) if val == postlocus_cue]
                    sentence = generated_post[sent_number]
                    if (locus_ambiguity in tokens and postlocus_cue in tokens and len(indices_locus_ambiguity) >= occurrence_locus_ambiguity and len(indices_postlocus_cue) >= occurrence_postlocus_cue):
                        interpretation, detailed = get_interpretation(tokens, parse, pos, locus_ambiguity, data = data_type, with_postlocus_cue= True,
                                                                        postlocus_cue = postlocus_cue, occurrence_locus_ambiguity= occurrence_locus_ambiguity,
                                                                  occurrence_postlocus_cue=occurrence_postlocus_cue, parser = parser)
                        output_file_post.write('\t'.join([item_id, sent_id,sentence, interpretation, detailed]) + '\n')
                    sent_number += 1
    output_file_pre.close()
    output_file_post.close()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--parser', action="store", default='allennlp') # allennlp

    arg_parser.add_argument('--lm', action="store", default= 'LSTM') # LSTM, GPT2

    arg_parser.add_argument('--ambiguity', action="store", default='NP-S') # ambiguity type (NP-S, NP-Z, N-V_V, N-V_N)

    args = arg_parser.parse_args()

    generated_dir = 'generated/' + args.ambiguity + '/' + args.lm + '/'

    parser = args.parser

    print(generated_dir)
    ambiguity = args.ambiguity

    if ambiguity.startswith('N-V'):
        category = ambiguity.split('_')[-1]
    else:
        category = None

    data_dir = 'data/'
    
    
    if not args.ambiguity == 'N-V':
        incorrect_ambig = evaluate_original_sentences(data_dir, data_type=ambiguity, parser=parser, ambig=True)
        incorrect_unambig = evaluate_original_sentences(data_dir, data_type=ambiguity, parser=parser, ambig=False)
        to_discard = list(set(incorrect_ambig).union(set(incorrect_unambig)))
        print('Discarded', len(to_discard), 'items') # Discarded: failed classification on original sentences
        dirs_to_evaluate = [generated_dir + i + '/' for i in os.listdir(generated_dir) if not 'original_sentences' in i and not i.startswith('.')]
        for dir in dirs_to_evaluate:
            evaluate_sentences(data_dir, dir, to_discard = to_discard,  data_type=ambiguity, parser=parser, ambig=True)
            evaluate_sentences(data_dir, dir, to_discard = to_discard,  data_type=ambiguity, parser=parser, ambig=False)
    else:
        to_discard = []
        for subtype in ['N-V_N', 'N-V_V']:
            df_ambig = pd.read_csv(data_dir + 'prompts_' + subtype + '_from_ambiguous.tsv', sep='\t')
            to_discard += list(df_ambig[df_ambig.disambiguation_type != 'agreement'].item) # Discarded: not disambiguated by agreement only
            df_unambig = pd.read_csv(data_dir + 'prompts_' + subtype + '_from_unambiguous.tsv', sep='\t')
            to_discard += list(df_unambig[df_unambig.disambiguation_type != 'agreement'].item)
        to_discard = set(to_discard)
        print('Discarded', len(to_discard), 'items')
        for subtype in ['N-V_N', 'N-V_V']:
            incorrect_ambig = evaluate_original_sentences(data_dir, data_type=subtype, parser=parser, ambig=True)
            incorrect_unambig = evaluate_original_sentences(data_dir, data_type=subtype, parser=parser, ambig=False)
            to_discard = to_discard.union(set(incorrect_ambig).union(set(incorrect_unambig))) # Discarded: failed classification on original sentences
        to_discard = list(to_discard)
        print('Discarded', len(to_discard), 'items')
        for subtype in ['N-V_N', 'N-V_V']:
            generated_dir_tmp  = generated_dir.replace(args.ambiguity, args.ambiguity + '/' + subtype)
            dirs_to_evaluate = [generated_dir_tmp + i + '/' for i in os.listdir(generated_dir_tmp) if not 'original_sentences' in i and not i.startswith('.')]
            for dir in dirs_to_evaluate:
                evaluate_sentences(data_dir, dir, to_discard = to_discard,  data_type=subtype, parser=parser, ambig=True)
                evaluate_sentences(data_dir, dir, to_discard = to_discard,  data_type=subtype, parser=parser, ambig=False)
        
