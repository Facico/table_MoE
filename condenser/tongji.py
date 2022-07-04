import json
import os
import csv
from re import I
from tqdm import tqdm
import unicodedata
import regex
import copy
import jsonlines

class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()

class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            pass
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)

def _normalize(text):
    return unicodedata.normalize("NFD", text)

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    # Answer is a list of possible strings
    text = tokenizer.tokenize(text).words(uncased=True)
    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i : i + len(single_answer)]:
                return True

    return False
if __name__ == '__main__':
    search_path = "/data1/fch123/UDT-QA/condenser/run.nq.text.table.teIn"
    doc_dir = "/data1/fch123/UDT-QA-data/downloads/data/text_raw_table_docs" #"/data/fanchenghao/text_docs" #"/data1/fch123/UDT-QA-data/downloads/data/text_raw_table_docs"
    qas_path = '/data1/fch123/DPR/downloads/data/retriever/qas/nq-test.csv'
    ans_out = "/data1/fch123/UDT-QA/condenser/bert_raw_table_text.out.json"
    qas_dict = {}
    with open(qas_path, 'r') as f:
        df = csv.reader(f, delimiter = '\t')
        id = 0
        print('deal with qas...')
        for row in tqdm(df):
            ans = eval(row[1])
            qas_dict[str(id)] = {'text': row[0], 'answers': ans}
            id += 1

    passage_dict = {}
    print('deal with passage...')
    hasans_rank = {}
    for i in os.listdir(doc_dir):
        file_x = os.path.join(doc_dir, i)
        with jsonlines.open(file_x, 'r') as f:
            for i in tqdm(f):
                passage_dict[i["docid"]] = i["text"] + " " + i['title']

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    with open(search_path, 'r') as f:
        for i in tqdm(f.readlines()):
            qas_id, _, docid, rank_id, score, _ = i.split(' ')
            if(int(rank_id) > 100): continue #break
            rank_id = int(rank_id) - 1
            if qas_id not in hasans_rank:
                hasans_rank[qas_id] = [0 for j in range(100)]
            yn = has_answer(qas_dict[qas_id]['answers'], passage_dict[docid], tokenizer)
            #print(qas_dict[qas_id]['answers'], passage_dict[docid])
            if yn == True:
                hasans_rank[qas_id][rank_id] = 1
            

    with open(ans_out, 'w') as f:
        json.dump(hasans_rank, f)

    AP_id = [1, 5, 10, 20, 50, 100]
    for j in AP_id:
        r = 0
        tot = 0
        for k, v in hasans_rank.items():
            yes = 0
            for i in range(j):
                if(v[i] == 1):
                    yes += 1
            if(yes > 0):
                r += 1
            tot += 1
        print('R@{}: {:.3f}'.format(j, r / tot))

    """data = json.load(open('run.nq.test.json', 'r'))
    AP_id = [1, 5, 10, 20, 50, 100]
    for j in AP_id:
        r = 0
        tot = 0
        for k, v in data.items():
            yes = 0
            for i in range(j):
                if(v["contexts"][i]["has_answer"] == True):
                    yes += 1
            if(yes > 0):
                r += 1
            tot += 1
        print('R@{}: {:.3f}'.format(j, r / tot))"""
            