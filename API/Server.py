from flask import Flask, request, jsonify
from sklearn.preprocessing import normalize
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter
import math
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import re
from konlpy.tag import Komoran

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# host: cmd --> ipconfig --> IP값
host="192.168.0.9"
@app.route("/", methods=['GET', 'POST', 'DELETE', 'PUT'])
def get():
    # JSONData: 앱으로부터 받은 JSON
    JSONData = request.get_json()

    #getData: 받은 JSON의 VALUE값
    getData = JSONData.get('contents')
    print(getData)

    ### 요약 알고리즘 ###
    def pagerank(x, df=0.85, max_iter=30, bias=None):  # x:정방매트릭스, df:다시검색할확률, m_i:최대실행횟수, bias:편차
        assert 0 < df < 1

        # 초기화
        A = normalize(x, axis=0, norm='l1')
        R = np.ones(A.shape[0]).reshape(-1, 1)

        # 편차확인
        if bias is None:
            bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
        else:
            bias = bias.reshape(-1, 1)
            bias = A.shape[0] * bias / bias.sum()
            assert bias.shape[0] == A.shape[0]
            bias = (1 - df) * bias

        # 실행횟수
        for _ in range(max_iter):
            R = df * (A * R) + bias

        return R  # 반환:벡터

    def scan_vocabulary(sents, tokenize=None, min_count=2):  # sents:문장변수, tokenize:토큰화변수, m_c:최소주기
        counter = Counter(w for sent in sents for w in tokenize(sent))
        counter = {w: c for w, c in counter.items() if c >= min_count}
        idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]  # 정렬된 단어
        vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}  # 정렬을 위한 사전
        return idx_to_vocab, vocab_to_idx

    def tokenize_sents(sents, tokenize):  # 토큰화 함수
        return [tokenize(sent) for sent in sents]  # 토큰화 된 리스트 반환

    def vectorize(tokens, vocab_to_idx):  # 벡터화 함수
        rows, cols, data = [], [], []
        for i, tokens_i in enumerate(tokens):
            for t, c in Counter(tokens_i).items():
                j = vocab_to_idx.get(t, -1)
                if j == -1:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(c)
        n_sents = len(tokens)
        n_terms = len(vocab_to_idx)
        x = csr_matrix((data, (rows, cols)), shape=(n_sents, n_terms))
        return x

    def sent_graph(sents, tokenize=None, min_count=2, min_sim=0.3,
                   similarity=None, vocab_to_idx=None, verbose=False):
        if vocab_to_idx is None:
            idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
        else:
            idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x: x[1])]

        x = vectorize_sents(sents, tokenize, vocab_to_idx)
        if similarity == 'cosine':
            x = numpy_cosine_similarity_matrix(x, min_sim, verbose, batch_size=1000)
        else:
            x = numpy_textrank_similarity_matrix(x, min_sim, verbose, batch_size=1000)
        return x

    def vectorize_sents(sents, tokenize, vocab_to_idx):
        rows, cols, data = [], [], []
        for i, sent in enumerate(sents):
            counter = Counter(tokenize(sent))
            for token, count in counter.items():
                j = vocab_to_idx.get(token, -1)
                if j == -1:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(count)
        n_rows = len(sents)
        n_cols = len(vocab_to_idx)
        return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    def numpy_cosine_similarity_matrix(x, min_sim=0.3, verbose=True, batch_size=1000):
        n_rows = x.shape[0]
        mat = []
        for bidx in range(math.ceil(n_rows / batch_size)):
            b = int(bidx * batch_size)
            e = min(n_rows, int((bidx + 1) * batch_size))
            psim = 1 - pairwise_distances(x[b:e], x, metric='cosine')
            rows, cols = np.where(psim >= min_sim)
            data = psim[rows, cols]
            mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))
            if verbose:
                print('\rcalculating cosine sentence similarity {} / {}'.format(b, n_rows), end='')
        mat = sp.sparse.vstack(mat)
        if verbose:
            print('\rcalculating cosine sentence similarity was done with {} sents'.format(n_rows))
        return mat

    def numpy_textrank_similarity_matrix(x, min_sim=0.3, verbose=True, min_length=1, batch_size=1000):
        n_rows, n_cols = x.shape

        # Boolean matrix
        rows, cols = x.nonzero()
        data = np.ones(rows.shape[0])
        z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

        # Inverse sentence length
        size = np.asarray(x.sum(axis=1)).reshape(-1)
        size[np.where(size <= min_length)] = 10000
        size = np.log(size)

        mat = []
        for bidx in range(math.ceil(n_rows / batch_size)):

            # slicing
            b = int(bidx * batch_size)
            e = min(n_rows, int((bidx + 1) * batch_size))

            # dot product
            inner = z[b:e, :] * z.transpose()

            # sentence len[i,j] = size[i] + size[j]
            norm = size[b:e].reshape(-1, 1) + size.reshape(1, -1)
            norm = norm ** (-1)
            norm[np.where(norm == np.inf)] = 0

            # normalize
            sim = inner.multiply(norm).tocsr()
            rows, cols = (sim >= min_sim).nonzero()
            data = np.asarray(sim[rows, cols]).reshape(-1)

            # append
            mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))

            if verbose:
                print('\rcalculating textrank sentence similarity {} / {}'.format(b, n_rows), end='')

        mat = sp.sparse.vstack(mat)
        if verbose:
            print('\rcalculating textrank sentence similarity was done with {} sents'.format(n_rows))

        return mat

    def graph_with_python_sim(tokens, verbose, similarity, min_sim):
        if similarity == 'cosine':
            similarity = cosine_sent_sim
        elif callable(similarity):
            similarity = similarity
        else:
            similarity = textrank_sent_sim

        rows, cols, data = [], [], []
        n_sents = len(tokens)
        for i, tokens_i in enumerate(tokens):
            if verbose and i % 1000 == 0:
                print('\rconstructing sentence graph {} / {} ...'.format(i, n_sents), end='')
            for j, tokens_j in enumerate(tokens):
                if i >= j:
                    continue
                sim = similarity(tokens_i, tokens_j)
                if sim < min_sim:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(sim)
        if verbose:
            print('\rconstructing sentence graph was constructed from {} sents'.format(n_sents))
        return csr_matrix((data, (rows, cols)), shape=(n_sents, n_sents))

    def textrank_sent_sim(s1, s2):
        n1 = len(s1)
        n2 = len(s2)
        if (n1 <= 1) or (n2 <= 1):
            return 0
        common = len(set(s1).intersection(set(s2)))
        base = math.log(n1) + math.log(n2)
        return common / base

    def cosine_sent_sim(s1, s2):
        if (not s1) or (not s2):
            return 0

        s1 = Counter(s1)
        s2 = Counter(s2)
        norm1 = math.sqrt(sum(v ** 2 for v in s1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in s2.values()))
        prod = 0
        for k, v in s1.items():
            prod += v * s2.get(k, 0)
        return prod / (norm1 * norm2)

    def word_graph(sents, tokenize=None, min_count=2, window=2,
                   min_cooccurrence=2, vocab_to_idx=None, verbose=False):
        if vocab_to_idx is None:
            idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
        else:
            idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x: x[1])]

        tokens = tokenize_sents(sents, tokenize)
        g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence, verbose)
        return g, idx_to_vocab

    def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2, verbose=False):
        counter = defaultdict(int)
        for s, tokens_i in enumerate(tokens):
            if verbose and s % 1000 == 0:
                print('\rword cooccurrence counting {}'.format(s), end='')
            vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
            n = len(vocabs)
            for i, v in enumerate(vocabs):
                if window <= 0:
                    b, e = 0, n
                else:
                    b = max(0, i - window)
                    e = min(i + window, n)
                for j in range(b, e):
                    if i == j:
                        continue
                    counter[(v, vocabs[j])] += 1
                    counter[(vocabs[j], v)] += 1
        counter = {k: v for k, v in counter.items() if v >= min_cooccurrence}
        n_vocabs = len(vocab_to_idx)
        if verbose:
            print('\rword cooccurrence counting from {} sents was done'.format(s + 1))
        return dict_to_mat(counter, n_vocabs, n_vocabs)

    def dict_to_mat(d, n_rows, n_cols):
        rows, cols, data = [], [], []
        for (i, j), v in d.items():
            rows.append(i)
            cols.append(j)
            data.append(v)
        return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    class KeywordSummarizer:
        def __init__(self, sents=None, tokenize=None, min_count=2,
                     window=-1, min_cooccurrence=2, vocab_to_idx=None,
                     df=0.85, max_iter=30, verbose=False):

            self.tokenize = tokenize
            self.min_count = min_count
            self.window = window
            self.min_cooccurrence = min_cooccurrence
            self.vocab_to_idx = vocab_to_idx
            self.df = df
            self.max_iter = max_iter
            self.verbose = verbose

            if sents is not None:
                self.train_textrank(sents)

        def train_textrank(self, sents, bias=None):
            g, self.idx_to_vocab = word_graph(sents,
                                              self.tokenize, self.min_count, self.window,
                                              self.min_cooccurrence, self.vocab_to_idx, self.verbose)
            self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
            if self.verbose:
                print('trained TextRank. n words = {}'.format(self.R.shape[0]))

        def keywords(self, topk=30):
            if not hasattr(self, 'R'):
                raise RuntimeError('Train textrank first or use summarize function')
            idxs = self.R.argsort()[-topk:]
            keywords = [(self.idx_to_vocab[idx], self.R[idx]) for idx in reversed(idxs)]
            return keywords

        def summarize(self, sents, topk=30):
            self.train_textrank(sents)
            return self.keywords(topk)

    class KeysentenceSummarizer:
        def __init__(self, sents=None, tokenize=None, min_count=2,
                     min_sim=0.3, similarity=None, vocab_to_idx=None,
                     df=0.85, max_iter=30, verbose=False):

            self.tokenize = tokenize
            self.min_count = min_count
            self.min_sim = min_sim
            self.similarity = similarity
            self.vocab_to_idx = vocab_to_idx
            self.df = df
            self.max_iter = max_iter
            self.verbose = verbose

            if sents is not None:
                self.train_textrank(sents)

        def train_textrank(self, sents, bias=None):
            g = sent_graph(sents, self.tokenize, self.min_count,
                           self.min_sim, self.similarity, self.vocab_to_idx, self.verbose)
            self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
            if self.verbose:
                print('trained TextRank. n sentences = {}'.format(self.R.shape[0]))

        def summarize(self, sents, topk=30, bias=None):
            n_sents = len(sents)
            if isinstance(bias, np.ndarray):
                if bias.shape != (n_sents,):
                    raise ValueError('The shape of bias must be (n_sents,) but {}'.format(bias.shape))
            elif bias is not None:
                raise ValueError('The type of bias must be None or numpy.ndarray but the type is {}'.format(type(bias)))

            self.train_textrank(sents, bias)
            idxs = self.R.argsort()[-topk:]
            keysents = [(idx, self.R[idx], sents[idx]) for idx in reversed(idxs)]
            return keysents

    # 위쪽은 textrank 알고리즘 코드

    # 밑은 기사 요약 부분

    # resp = requests.get("https://sports.news.naver.com/news.nhn?oid=108&aid=0002884817")  # 기사 링크
    # soup = bs(resp.text, "html.parser")  # html 소스코드 전체
    # soup_body = str(soup.find("div", {"id": "newsEndContents"}))  # div태그에서 id로 기사 내용에 해당하는 부분 soup_body에 저장
    soup_body = getData

    # 기사 파싱(다듬기)
    soup_body = re.sub('<.+?>', '', soup_body, 0).strip()
    soup_body = re.sub('\t', '', soup_body, 0).strip()
    soup_body = re.sub('\n', '', soup_body, 0).strip()
    soup_body = re.sub('/사진=', '', soup_body, 0).strip()
    soup_body = re.sub('&gt', '', soup_body, 0).strip()
    soup_body = re.sub('&lt', '', soup_body, 0).strip()
    sents = soup_body.split(". ")  # sents는 마침표로 나뉜 문자열 리스트
    # 기사 파싱(다듬기) - end

    komoran = Komoran()  # 단어 단위로 쪼개기

    def komoran_tokenizer(sent):
        words = komoran.pos(sent, join=True)
        words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words

    summarizer = KeysentenceSummarizer(
        tokenize=komoran_tokenizer,
        min_sim=0.3,
        verbose=False
    )  # 요약기

    keysents = summarizer.summarize(sents, topk=3)  # 요약된 세 문장이 들어있는 리스트(가중치 포함)

    print(keysents)

    # data: 앱에 전송 할 데이터
    # 요약 알고리즘을 통해 얻은 값을 data에 입력
    data = ""
    for i in range(0, 3):
        data += keysents[i][2] + ". "

    print(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host=host)