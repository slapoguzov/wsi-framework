#!/usr/bin/env python

from __future__ import print_function
import argparse
from io import TextIOWrapper

from pandas import read_csv

from quality import calculate_quality
from wsi import RandomWsi
from wsi import WordSenseInduction, Word


def evaluate(dataset_fpath: TextIOWrapper, output: TextIOWrapper, senseResolver: WordSenseInduction):
    df = read_csv(dataset_fpath, sep='\t', dtype={'gold_sense_id': str, 'predict_sense_id': str})
    for context_id, word, positions, text in list(zip(df.context_id, df.word, df.positions, df.context)):
        for position in positions.split(','):
            start, end = position.split('-')
            sense = senseResolver.resolve(Word(word, start, end), text)
            df.loc[df.context_id == context_id, 'predict_sense_id'] = sense.id

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
    evaluate(args.dataset, args.output, RandomWsi())
    calculate_quality(open(args.output.name, 'r', encoding='utf-8'))


if __name__ == '__main__':
    main()
