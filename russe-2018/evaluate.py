#!/usr/bin/env python

from __future__ import print_function

import argparse
from collections import Counter
from io import TextIOWrapper
from typing import List, Dict

from pandas import read_csv

from bert.simple_bert_embeddings import SimpleBertEmbeddings
from caching_word_embeddings import CachingWordEmbeddings
from clustering.affinity_propagation_clustering import AffinityPropagationClustering
from quality import calculate_quality
from wsc import WordSenseClustering
from wsi import Word
from wsi_wsc import WsiBasedWsc


def evaluate(dataset_fpath: TextIOWrapper, output: TextIOWrapper, sense_resolver: WsiBasedWsc):
    df = read_csv(dataset_fpath, sep='\t', dtype={'gold_sense_id': str, 'predict_sense_id': str})
    word_usages: Dict[str, List[Word]] = {}
    for context_id, word, positions, text in list(zip(df.context_id, df.word, df.positions, df.context)):
        word_usages[text] = []
        for position in positions.split(','):
            start, end = position.split('-')
            word_usages[text].append(Word(word, int(start), int(end)))

    sense_resolver.fit(WordSenseClustering(word_usages=word_usages,
                                           word_embeddings=CachingWordEmbeddings(SimpleBertEmbeddings('data/bert_rus')),
                                           vectors_clustering=AffinityPropagationClustering()))

    for context_id, word, positions, text in list(zip(df.context_id, df.word, df.positions, df.context)):
        senses: List[int] = []
        for position in positions.split(','):
            start, end = position.split('-')
            senses.append(sense_resolver.resolve(Word(word, int(start), int(end)), text).id)
        mode_sense = Counter(senses).most_common(1)[0][0]
        print("[evaluate] senses for", word, "=", senses, " mode =", mode_sense)
        df.loc[df.context_id == context_id, 'predict_sense_id'] = mode_sense
    df.to_csv(output, sep='\t', encoding='utf-8', index=False, line_terminator='\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=argparse.FileType('r', encoding='utf-8'),
                        help='Path to a CSV file with the dataset in the format '
                             '"context_id<TAB>word<TAB>gold_sense_id<TAB>predict_sense_id'
                             '<TAB>positions<TAB>context". ')
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('rw', encoding='utf-8'),
                        default=open('output.csv', mode='w+', encoding='utf-8'),
                        help='Path to a result file')
    args = parser.parse_args()
    evaluate(args.dataset, args.output, WsiBasedWsc())
    args.output.flush()
    calculate_quality(open(args.output.name, 'r', encoding='utf-8'))


if __name__ == '__main__':
    main()
