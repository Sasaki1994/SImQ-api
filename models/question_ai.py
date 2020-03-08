import numpy as np
from gensim import matutils
from gensim import corpora
import MeCab
import pickle
import os
import sys


class QuestionAI:
    """
    Bag-of-Wordsを利用した類似質問判定/レスポンス返却モデル

    variables:
        mecab(Mecab Tagger):分かち書きに利用するmecab
        dictionary(gensim dictionary):判定に利用する名詞の辞書
        input_data(dict of arrays):過去質問データ
        dist_matrix(numpy ndarray):過去質問データの分散表現行列
    """

    def __init__(self, base_dict='/var/lib/mecab/dic/mecab-ipadic-neologd'):
        self.mecab = MeCab.Tagger("-d {}".format(base_dict)) if base_dict else MeCab.Tagger("")
        self.mecab.parse("")
        self.dictionary = None
        self.input_data = {"curriculum_id": [], "text": [], "response": []}
        self.dist_matrix = None

    def learn_words(self, texts, no_below=0):
        """
        AIの辞書作成
        in: texts(list of string):質問テキスト配列
            no_below(int):覚えるべき単語の最低出現数
        """
        wakati_docs = []
        for text in texts:
            wakati_words = self._split_words(text)
            wakati_docs.append(wakati_words)
        dictionary = corpora.Dictionary(wakati_docs)
        dictionary.filter_extremes(no_below=no_below)
        self.dictionary = dictionary
        return self

    def fit(self, curriculum_ids, texts, responses):
        """
        質問データを学習. 予めlearn_wordsメソッドによる辞書登録が必要
        in:    curriculum_ids(np.array of string or int):質問箇所のカリキュラムID
               texts(np.array of string):質問本文
               responses(np.array of string):ユーザーに返す文章
        """
        if self.dictionary is None:
            print('Error: AI has no dictionary. Please call method "learn_words" ', file=sys.stderr)
            raise AttributeError

        if not (len(curriculum_ids) == len(texts) and len(curriculum_ids) == len(responses)):
            print('Error:  input data is not same size.', file=sys.stderr)
            raise IndexError

        # 分散表現行列の生成
        dist_matrix = np.empty((0, len(self.dictionary)), int)
        for text in texts:
            dist_matrix = np.append(dist_matrix, [self._trans2distvec(text)], axis=0)

        # データをモデルに格納
        curriculum_ids = [str(curriculum_id) for curriculum_id in curriculum_ids]
        self.input_data["curriculum_id"] = np.array(curriculum_ids)
        self.input_data["text"] = np.array(texts)
        self.input_data["response"] = np.array(responses)
        self.dist_matrix = dist_matrix

        return self

    def predict(self, curriculum_id, text, n=5):
        """
        類似質問を判定し、対応するレスポンスを返す                                                                                                                           nseを返す. 予めfitが必要.
        in:   curriculum_id(string or int):質問箇所のカリキュラムID
              text(string):質問本文
        out: responses(list of string):類似質問に対応したレスポンス.
        """

        if self.dist_matrix is None or self.dictionary is None:
            print('Error:  AI has not learned yet. Please call method "fit" beforehand. ', file=sys.stderr)
            raise AttributeError

        dist_vec = self._trans2distvec(text)

        # 同じカリキュラム箇所の質問を抽出
        curriculum_id = str(curriculum_id)
        same_curs_res_data = self.input_data["response"][self.input_data["curriculum_id"] == curriculum_id]
        same_curs_dist_mat = self.dist_matrix[self.input_data["curriculum_id"] == curriculum_id]

        # 　類似度の高いn個のデータインデックスを抽出
        sim_index = self._extract_sims(dist_vec, same_curs_dist_mat, n=n) if len(same_curs_dist_mat) > 0 else []

        # n個のレスポンス内容を格納して返却
        responses = []
        for i in sim_index:
            responses.append(same_curs_res_data[i])
        return responses

    def save_pickle(self, filename):
        """
        インスタンスのpickle保存メソッド
        :param filename:
        :return: None
        """
        tmp = self.mecab
        self.mecab = ""
        with open(filename, mode='wb') as f:
            pickle.dump(self, f)
        self.mecab = tmp
        return os.path.abspath(filename)

    @classmethod
    def load(cls, filename, base_dict='/var/lib/mecab/dic/mecab-ipadic-neologd'):
        """
        インスタンスのpickle読み込みメソッド
        :param filename:
        :param base_dict:
        :return: QuestionAI
        """
        with open(filename, mode='rb') as f:
            instance = pickle.load(f)
        instance.mecab = MeCab.Tagger("-d {}".format(base_dict)) if base_dict else MeCab.Tagger("")
        instance.mecab.parse("")
        return instance

    # private methods
    def _trans2distvec(self, text):
        """
        テキストを分散表現へ変換
        in:    text(string):質問本文
        out:  dist_vector(numpy array):質問の分散表現ベクトル(行ベクトル)
        """
        wakati_doc = self._split_words(text)
        bow = self.dictionary.doc2bow(wakati_doc)
        dense = matutils.corpus2dense([bow], num_terms=len(self.dictionary)).T[0]
        dist_vector = np.array(dense)
        return dist_vector

    def _extract_sims(self, dist_vec, dist_matrix, n=3):
        """
        類似質問の抽出
        in:    dist_vector(numpy array):質問の分散表現ベクトル
               n(int):抽出数
        out: sim_index(list of int):類似質問のインデックス配列
        """
        # コサイン類似度を用いた類似度計算
        sim_vector = self._cos_sim(dist_vec, dist_matrix)

        sim_index = []
        for _ in range(n):
            max_index = np.argmax(sim_vector)
            if sim_vector[max_index] == 0:
                break
            sim_index.append(max_index)
            sim_vector[max_index] = 0  # 抽出後の置き換え
        return sim_index

    def _cos_sim(self, vector, matrix):
        """
        コサイン類似度計算
        in:   vector(numpy array):n次元 行ベクトル
              matrix(numpy ndarray):m行n列行列
        out: sim_vector(numpy array):n次元のコサイン類似度を表す行ベクトル
        """
        # 大小関係に影響しない部分をコメントアウト
        numer = matrix @ vector.T
        mat_dinom = np.diag(matrix @ matrix.T)  # ** .5
        # vec_dinom = (vector @ vector.T)   ** .5
        return numer / (mat_dinom + 10**(-8))  # / vec_dinom

    def _split_words(self, text):
        """
        質問本文→名詞配列
        ex) 「chatspaceでDBに画像が保存されない」→ ['chatspace', 'DB', '画像', '保存']
        in:    text(stirng):質問本文
        out:  wakati_words(list):名詞配列
        """
        # neologd使用の場合、必要
        if type(text) is not str:
            text = ""

        # mecabで分かち書き
        node = self.mecab.parseToNode(text)

        # 名詞のみ抽出
        wakati_words = []
        while node is not None:
            hinsi = node.feature.split(",")[0]
            if hinsi in ["名詞"]:
                wakati_words.append(node.surface)
            node = node.next
        return wakati_words


if __name__ == '__main__':
    pass
