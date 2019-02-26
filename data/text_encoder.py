import re
import json

import ftfy
import spacy


class TextEncoder(object):

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.start_token_ = '_start_'
        self.delimeter_token_ = '_delimiter_'
        self.end_token_ = '_end_'
        self.encoder[self.start_token_] = len(self.encoder)
        self.encoder[self.delimeter_token_] = len(self.encoder)
        self.encoder[self.end_token_] = len(self.encoder)
        self.start_token = self.encoder[self.start_token_]
        self.delimeter_token = self.encoder[self.delimeter_token_]
        self.end_token = self.encoder[self.end_token_]
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.end_of_word = '</w>'

    def get_pairs(self, word):
        return set([(word[i], word[i + 1]) for i in range(len(word)) if i + 1 < len(word)])

    def standardize_text(self, text):
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
        text = re.sub(r'\s*\n\s*', ' \n ', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
        return text.strip()

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        encoded_word = tuple(token[:-1]) + ( token[-1] + self.end_of_word,)
        pairs = self.get_pairs(encoded_word)
        if not pairs:
            return token + self.end_of_word
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_encoded_word = []
            i = 0
            while i < len(encoded_word):
                try:
                    j = encoded_word.index(first, i)
                    new_encoded_word.extend(encoded_word[i:j])
                    i = j
                except:
                    new_encoded_word.extend(encoded_word[i:])
                    break
                if encoded_word[i] == first and i < len(encoded_word)-1 and encoded_word[i+1] == second:
                    new_encoded_word.append(first+second)
                    i += 2
                else:
                    new_encoded_word.append(encoded_word[i])
                    i += 1
            encoded_word = tuple(new_encoded_word)
            if len(encoded_word) == 1:
                break
            else:
                pairs = self.get_pairs(encoded_word)
        encoded_word = ' '.join(encoded_word)
        if encoded_word == '\n  {}'.format(self.end_of_word):
            encoded_word = '\n{}'.format(self.end_of_word)
        self.cache[token] = encoded_word
        return encoded_word

    def encode(self, document):
        document = self.nlp(self.standardize_text(ftfy.fix_text(document)))
        document_tokens = []
        for token in document:
            document_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
        return document_tokens
