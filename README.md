
# The language model understood the prompt was ambiguous: probing syntactic uncertainty through generation

Code and data relevant to the experiments in "The language model understood the prompt was ambiguous: 
probing syntactic uncertainty through generation", to appear in Proc. BlackboxNLP 2021


## Content

- `data/`: prompts from sentence pairs (see `data/README.md`); NP/S, NP/Z data from [Grodner et al., (2003)](https://link.springer.com/article/10.1023/A:1022496223965); Noun/Verb data from [Frazier and Rayner, (1986)](https://www.sciencedirect.com/science/article/abs/pii/0749596X87901379)
- `src/`: code to generate completions from pairs, parse them and extract interpretations

## Instructions

To generate from prompts, e.g., with GPT2 on NP/S data 

    python generate_from_prompts.py --lm GPT2 --ambiguity NP-S 

By default using standard sampling. To generate with beam search and exploring parameters in stochastic decoding:

    python generate_from_prompts.py --lm GPT2 --ambiguity NP-S --all --search

To assign interpretations the generated sentences:

    python parse_sentences.py --lm GPT2 --ambiguity NP-S
    python extract_interpretations.py --lm GPT2 --ambiguity NP-S

--lm : `GPT2`, `LSTM`
--ambiguity: `NP-S`, `NP-Z`, `N-V`

Running these commands outputs files with the generated sentences, their parses and interpretation in a directory e.g., `generated/NP-S/GPT2/GPT2/sampling-p_p1_temperature1_repetition1`, containing:
- for each prompt, files with generated sentences, e.g., `NP-S_1_nocue.tsv`
-`parses_allennlp/`:  for each prompt, parses and PoS labels of generated sentences
- `ìnterpretations_allennlp`:  for each prompt type, all the generated sentences with their associated interpretations

To use and run the LSTM model from [Gulordava et al., (2018)](https://www.aclweb.org/anthology/N18-1108/):
- Download the English model and vocab file from <https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/data>
- Rename the model as the `model.pt` and the vocal file as `vocab.txt`; place the files in a new directory `LSTM/`
- Download `model.py` and `utils.py` from <https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/src/language_models> ; place these files in `src/`.
To generate from the LSTM model: PyTorch <=1.2
