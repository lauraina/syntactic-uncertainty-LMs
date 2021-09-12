
## Prompts data

Prompts from sentence pairs: NP/S, NP/Z data from [Grodner et al., (2003)](https://link.springer.com/article/10.1023/A:1022496223965); Noun/Verb data from [Frazier and Rayner, (1986)](https://www.sciencedirect.com/science/article/abs/pii/0749596X87901379)

For each ambiguity:
4 prompts are derived from a sentence pair made out of an ambiguous and unambiguous sentence:
- `from_ambiguous`  (without pre-locus cue): no-locus cue prompt (`prompt_pre`), post-locus cue prompt (`prompt_post`)
- `unambiguous` (with pre-locus cue) : pre-locus cue prompt (`prompt_pre`), pre&post-locus cue prompt (`prompt_post`)


`NP-S`: NP/S ambiguity data
`NP-Z`: NP/Z ambiguity data
`N-V_V`: Noun/Verb ambiguity data with prompts disambiguating the locus of ambiguity as Verb
`N-V_N`: Noun/Verb ambiguity data with prompts disambiguating the locus of ambiguity as Noun


In each `.tsv` file, for each row:
- `item`: identifier of sentence
- `sentence` and `sentence_original` (the first includes modifications to words if not covered by the LSTM vocabulary; same for prompts)
- `locus_ambiguity`
- `postlocus_cue`
- `prompt_pre`, `prompt_post`, `prompt_pre_original`, `prompt_post_original`
- `occurrence_postlocus_cue`: in case the post-locus cue word appears multiple times in the sentence, number of the occurrence that is the post-locus cue
- `occurrence_locus_ambiguity`: in case the locus of ambiguity word appears multiple times in the sentence, number of the occurrence that is the locus of ambiguity
- For N-V_V, N-V_N: `disambiguation_type`: `agreement` (change in determiner) and/or `change_in_modifier` (only agreement is kept for the analyses)

