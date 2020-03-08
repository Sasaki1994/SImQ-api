import pytest

import numpy as np
from gensim import corpora
from question_ai import QuestionAI
import MeCab
import os

# Unit Tests
class TestQuestionAI:

    test_ai = QuestionAI()

    @pytest.fixture()
    def set_dictionary(self):
        wakati_docs = [['chatspace', '画像', '保存'],
                       ['chatspace', 'ビュー', '画像', 'アイコン'],
                       ['erb', '2', 'haml', 'エラー'],
                       ['rails', 'db:migrate', 'ところ', 'sql', 'エラー'],
                       ['haml', 'エラー']
                       ]
        self.test_ai.dictionary = corpora.Dictionary(wakati_docs)
        yield
        self.test_ai.__init__()

    @pytest.fixture()
    def fitted(self):
        curriculum_ids = ['3329', '3329', '3329', '2122', '1124']
        texts = ['chatspaceに画像がうまく保存されない',
                 'chatspaceのビューが作れない。画像アイコンができない',
                 'erb2hamlを導入したがエラーが出て進めない',
                 'rails db:migrateを実行したところ、sqlエラーが出る',
                 'hamlでエラーが出る']
        responses = ['Answer1', 'rails s\n動く', 'hamlでビューを作ったが動かない', 'Ans', 'Ans']
        self.test_ai.fit(curriculum_ids, texts, responses)
        yield
        self.test_ai.__init__()

    @pytest.fixture()
    def set_dist_matrix(self):
        dist_matrix = np.array([[0, 1, 3],
                               [3, 0, 0],
                               [1, 1, 1],
                               [0, 0, 0],
                               [0, 0, 0]])
        return dist_matrix

    def test_init_mecab(self):
        assert isinstance(self.test_ai.mecab, MeCab.Tagger)

    def test_learn_words_normal(self):
        texts = ['chatspaceに画像がうまく保存されない',
                 'chatspaceのビューが作れない。画像アイコンができない',
                 'erb2hamlを導入したがエラーが出て進めない',
                 'rails db:migrateを実行したところ、sqlエラーが出る',
                 'hamlでエラーが出る'
                 ]
        self.test_ai.learn_words(texts)
        assert isinstance(self.test_ai.dictionary, corpora.Dictionary) and len(self.test_ai.dictionary) > 0

    def test_learn_words_below(self):
        texts = ['chatspaceに画像がうまく保存されない',
                 'chatspaceのビューが作れない。\n画像アイコンができない',
                 'erb2hamlを導入したがエラーが出て進めない',
                 'rails db:migrateを実行したところ、sqlエラーが出る',
                 'hamlでエラーが出る'
                 ]
        dict_0 = self.test_ai.learn_words(texts, no_below=0).dictionary
        dict_2 = self.test_ai.learn_words(texts, no_below=2).dictionary
        dict_4 = self.test_ai.learn_words(texts, no_below=4).dictionary
        assert (len(dict_0) > len(dict_2)) and (len(dict_2) > len(dict_4))

    def test_fit_dist_matrix(self, set_dictionary):
        curriculum_ids = ['3329', '2122', '3329']
        texts = ['hamlの書き方が分からない', 'rails sが\n動かない', 'hamlでビューを作ったが動かない']
        responses = ['Answer1', 'rails s\n動く', 'hamlでビューを作ったが動かない']
        self.test_ai.fit(curriculum_ids, texts, responses)
        dist_matrix = self.test_ai.dist_matrix
        assert dist_matrix.shape == (len(curriculum_ids), len(self.test_ai.dictionary))

    def test_fit_input_data(self, set_dictionary):
        curriculum_ids = [3329, '2122', '最終課題']
        texts = ['hamlの書き方が分からない', 'rails sが\n動かない', 'hamlでビューを作ったが動かない']
        responses = ['Answer1', 'rails s\n動く', 'hamlでビューを作ったが動かない']
        self.test_ai.fit(curriculum_ids, texts, responses)
        input_data = self.test_ai.input_data
        is_same_len1 = len(input_data['curriculum_id']) == len(input_data['text'])
        is_same_len2 = len(input_data['text']) == len(input_data['response'])
        assert is_same_len1 and is_same_len2

    def test_fit_without_dict(self):
        self.test_ai = QuestionAI()
        with pytest.raises(AttributeError):
            curriculum_ids = ['3329', '2122', '最終課題']
            texts = ['hamlの書き方が分からない', 'rails sが\n動かない', 'hamlでビューを作ったが動かない']
            responses = ['Answer1', 'rails s\n動く', 'hamlでビューを作ったが動かない']
            self.test_ai.fit(curriculum_ids, texts, responses)

    @pytest.mark.parametrize('curs, txts, reses', [
        (['3329'], ['haml', 'ビュー'], ['Ans1', 'Ans2']),
        (['3329', '1211'], ['haml'], ['Ans1', 'Ans2']),
        (['3329', '1211'], ['haml', 'ビュー'], ['Ans1'])
    ])
    def test_fit_without_dict(self, set_dictionary, curs, txts, reses):
        with pytest.raises(IndexError):
            self.test_ai.fit(curs, txts, reses)

    @pytest.mark.parametrize('n, cur_id', [
        (3, '3329'),
        (3, 3329),
        (2, '3329'),
        (5, '3329'),
        (3, 2122),
        (3, 1419)
    ])
    def test_predict(self, set_dictionary, fitted, n, cur_id):
        text = 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'
        predicts = self.test_ai.predict(cur_id, text, n=n)
        isin_list = [predict in self.test_ai.input_data['response'] for predict in predicts]
        assert all(isin_list)

    def test_predict_without_dictionary(self):
        cur_id = '3329'
        text = 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'
        with pytest.raises(AttributeError):
            self.test_ai.predict(cur_id, text)

    def test_predict_without_fit(self, set_dictionary):
        cur_id = '3329'
        text = 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'
        with pytest.raises(AttributeError):
            self.test_ai.predict(cur_id, text)

    def test_save(self, set_dictionary, fitted, tmpdir):
        path = self.test_ai.save_pickle(tmpdir.join("save.pickle"))
        assert os.path.exists(path)

    def test_load(self, set_dictionary, fitted, tmpdir):
        test_ai = self.test_ai
        path = test_ai.save_pickle(tmpdir.join("save.pickle"))
        load_ai = QuestionAI.load(path, base_dict="")
        load_ai2 = QuestionAI.load(path)
        assert load_ai.dist_matrix is not None and load_ai2.dist_matrix is not None

    # private
    def test_trans2distvec(self, set_dictionary):
        """
        GIVEN: learn_wordsでdictionaryセット済みならば
        WHEN : textを与えると
        THEN : 分散表現ベクトルが得られる
        """
        text = "応用を進めていたところ、hamlの書き方が分からなくなりエラーが修正できない"
        dist_vector = self.test_ai._trans2distvec(text)
        assert len(dist_vector) == len(self.test_ai.dictionary)

    def test_trans2distvec_val(self, set_dictionary):
        text = "応用を進めていたところ、hamlの書き方が分からなくなりエラーが修正できない"
        dist_vector = self.test_ai._trans2distvec(text)
        assert any(dist_vector != np.zeros(len(self.test_ai.dictionary)))

    def test_extract_sims_index(self, set_dist_matrix):
        dist_vec = np.array([2, 0, 0])
        sim_index = self.test_ai._extract_sims(dist_vec, set_dist_matrix)
        ex_index = [1, 2, 0]
        assert sim_index == ex_index

    def test_extract_sims_index(self, set_dist_matrix):
        dist_vec = np.array([0, 0, 0])
        sim_index = self.test_ai._extract_sims(dist_vec, set_dist_matrix)
        assert len(sim_index) == 0

    def test_cos_sim(self):
        vector = np.array([1, 2, 3])
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        cos_sim = self.test_ai._cos_sim(vector, matrix)
        ex_cos_sim = np.array([14, 32])/np.array([14, 77])
        # 0割対策の10**(-8)の誤差を許容する
        assert all(cos_sim <= ex_cos_sim) and all(cos_sim >= ex_cos_sim - 10**(-4))

    def test_cos_sim_zero(self):
        vector = np.array([0, 0, 0])
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        cos_sim = self.test_ai._cos_sim(vector, matrix)
        assert all(cos_sim == np.zeros(2))

    def test_split_words(self):
        wakati_words = self.test_ai._split_words('chatspaceで画像が保存されない')
        assert wakati_words == ['chatspace', '画像', '保存']


@pytest.mark.parametrize('cur_id, text', [
        ('3329', 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'),
        (3329, 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'),
        ('3329', 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'),
        ('3329', 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'),
        (2122, 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。'),
        (1419, 'hamlについて詳しく聞きたい。\nビューの作り方が分からない。')
    ])
def test_join(cur_id, text):
    join_test_ai = QuestionAI()
    curriculum_ids = ['3329', '3329', '3329', '2122', '1124']
    texts = ['chatspaceに画像がうまく保存されない',
             'chatspaceのビューが作れない。画像アイコンができない',
             'erb2hamlを導入したがエラーが出て進めない',
             'rails db:migrateを実行したところ、sqlエラーが出る',
             'hamlでエラーが出る']
    responses = ['Answer1', 'rails s\n動く', 'hamlでビューを作ったが動かない', 'Ans', 'Ans']

    join_test_ai.learn_words(texts)
    join_test_ai.fit(curriculum_ids, texts, responses)
    response = join_test_ai.predict(cur_id, text)
    print(response)
    assert response is not None



