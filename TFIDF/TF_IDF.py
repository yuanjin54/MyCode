from gensim import corpora, models, similarities
import time
import os
from functools import wraps
import jieba.posseg as posseg


def is_chinese(word):
    '''判断一个词语是不是中文'''
    for ch in list(word):
        if ch < u'\u4e00' or ch > u'\u9fff':
            return False
    return True


def decorator(func):
    '''这是一个监控每个方法执行时间的探测器'''

    @wraps(func)
    def wrap():
        start_time = time.time()
        print('input {} function，start time：{}'.format(func.__name__, start_time))
        func()
        print('{} function run end, spends：{} second'.format(func.__name__, str(time.time() - start_time)))
        print('*' * 30 + ' spilt line ' + '*' * 30)

    return wrap


# LSI模型
def lsi():
    words, names = get_all_words()
    dictionary = corpora.Dictionary(words)
    doc_vectors = [dictionary.doc2bow(word) for word in words]
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]

    lsi_model = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=2)

    query = get_words(QUERY_DOCUMENT_PATH)
    query_bow = dictionary.doc2bow(query)
    query_lsi = lsi_model[query_bow]

    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_lsi]
    print("====" * 10)
    print(names)
    print(list(enumerate(sims)))
    print("*" * 20 + " end " + "*" * 20)


# tf-idf模型
@decorator
def tf_idf():
    words, names = get_all_words()
    # print("words size: {}".format(len(words)))
    dictionary = corpora.Dictionary(words)  # 相当于给每个单词生成一个唯一的id
    print(dictionary.token2id)
    print(dictionary.dfs)  # 统计每个单词出现的次数
    doc_vectors = [dictionary.doc2bow(word) for word in words]  # 生成词袋，例如：(8, 2)表示id=8的单次在文章中出现了2次
    print(doc_vectors)

    tfidf = models.TfidfModel(doc_vectors)  # 建立 TFIDF 模型
    tfidf_vectors = tfidf[doc_vectors]

    query = get_words(QUERY_DOCUMENT_PATH)  # 需要找出查询的文章
    query_bow = dictionary.doc2bow(query)  # 生成词袋

    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_bow]  # 得到与其他文章的相似度
    print("====" * 10)
    print(names)
    print(list(enumerate(sims)))
    print("*" * 20 + " end " + "*" * 20)


def get_all_words():
    '''获取所有文章的词语, 每篇文章是一个list, list的词语是去除停用词后的词语'''
    all_words = []
    filenames = []
    for parent, dir_names, file_names in os.walk(DOCUMENT_PATH, followlinks=True):
        for filename in file_names:
            file_path = os.path.join(parent, filename)
            all_words.append(get_words(file_path))
            filenames.append(filename)
    return all_words, filenames


def get_words(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        all_words = posseg.cut(content)
    for word, flag in all_words:
        if is_chinese(word) and flag not in STOP_FLAG:
            words.append(word)
    return words


def main():
    print("main")


if __name__ == '__main__':
    # 下面这些词性的词被分为停用词
    STOP_FLAG = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    STOP_WORDS_PATH = 'stopWords'
    DOCUMENT_PATH = 'documents'
    QUERY_DOCUMENT_PATH = 'documents/老年高血压'
    tf_idf()
    lsi()
