import os
import numpy
import click
import gensim
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
from transformers import AlbertTokenizer, TFAlbertModel


def word2vec(input_file_name, output_dir, embedding_dim, window, iter, min_count):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    model = gensim.models.Word2Vec(corpus_file=input_file_name,
                                   window=window,
                                   size=embedding_dim,
                                   iter=iter,
                                   min_count=min_count)

    model.save(output_dir + "/word2vec.model")
    model.wv.save_word2vec_format(output_dir + "/word2vec_pattern.vec")


def create_embedding_matrix(input_file_name, word_index, vocab_size, embedding_dim, act_range=10000000):
    # Adding again 1 because of reserved 0 index
    vocab_size = vocab_size + 1
    embedding_matrix = numpy.zeros((vocab_size, embedding_dim))
    embedding_matrix[0].fill(-1.0)

    f = open(input_file_name, encoding='utf8', errors='ignore')
    max_range = int(f.readline().split(" ")[0])
    if(act_range > max_range):
        act_range = max_range

    index = 0
    with click.progressbar(length=act_range, label="LOAD EMBEDDING MATRIX: ", fill_char=click.style('=', fg='white')) as bar:
        while index < act_range:
            word, *vector = f.readline().split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = numpy.array(vector,
                                                    dtype=numpy.float32)[:embedding_dim]

            index = index + 1
            bar.update(1)

    f.close()

    return embedding_matrix


def load_embedding_space(input_file_name, embedding_dim, tokenizer):
    # Load embedding space
    vocab_size = len(tokenizer.word_index)
    embedding_matrix = create_embedding_matrix(input_file_name,
                                               tokenizer.word_index,
                                               vocab_size,
                                               embedding_dim,
                                               500000)

    nonzero_elements = numpy.count_nonzero(numpy.count_nonzero(embedding_matrix,
                                                               axis=1))
    print("\nvocabulary is covered by the pretrained model: ".upper() +
          str(round(nonzero_elements / vocab_size * 100, 2)) + "%")
    print("LOADING EMBEDDING SPACE: COMPLETE")

    return embedding_matrix


def save_word2vec(path, index_word, embedding_matrix_new):
    with open(path, "w", encoding='utf8', errors='ignore') as f:
        for i in range(1, len(embedding_matrix_new)-1):
            item = embedding_matrix_new[i]
            vec = "\t".join(map(str, item))
            tmp = vec + "\n"
            f.write(tmp)


def doc2vec_with_transformer(data, pretrained):
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = TFBertModel.from_pretrained(pretrained)

    vectors = []
    lonnger_conuter = 0
    with click.progressbar(data, label="SENTENCE EMBEDDING: ", fill_char=click.style('=', fg='white')) as data_bar:
        for doc in data_bar:
            sentences = doc.split(". ")
            sentences_vectors = tf.constant([[0]*768]).numpy()
            for sentence in sentences:
                tokenized_sentence = tokenizer.encode(sentence)
                tokenized_sentence = tokenized_sentence[:512]
                input_ids = tf.constant(tokenized_sentence)[None, :]
                outputs = model(input_ids)
                sentence_vectors = outputs[0][0].numpy()
                sentences_vectors = numpy.concatenate(
                    (sentences_vectors, sentence_vectors), axis=0)

            vectors.append(tf.reduce_mean(sentences_vectors, 0).numpy())

    return vectors


def tokenizer_test_and_train(train, test, num_words=100000, pad_maxlen_cover=0.99, pad_maxlen_limit=10000):
    embedded_vectors = []
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train)
    train_feauters = tokenizer.texts_to_sequences(train)
    test_feauters = tokenizer.texts_to_sequences(test)
    vocab_size = len(tokenizer.word_index) + 1

    pad_maxlen_lens = [len(item) for item in train_feauters]
    pad_maxlen_lens.sort()
    if pad_maxlen_cover <= 1 and pad_maxlen_cover >= 0 and isinstance(pad_maxlen_cover, float):
        max_element_index = int(len(pad_maxlen_lens) * pad_maxlen_cover) - 1
        pad_maxlen = pad_maxlen_lens[max_element_index]    
    else:
        pad_maxlen = pad_maxlen_cover
        
    pad_maxlen_max = max(pad_maxlen_lens)
    pad_maxlen_min = min(pad_maxlen_lens)
    if pad_maxlen_limit <= pad_maxlen_max:
        pad_maxlen = pad_maxlen_limit 

    train_feauters = pad_sequences(train_feauters,
                                   padding='post',
                                   maxlen=pad_maxlen)

    test_feauters = pad_sequences(test_feauters,
                                  padding='post',
                                  maxlen=pad_maxlen)

    result = {
        "train_feauters": train_feauters,
        "test_feauters": test_feauters,
        "vocab_size": vocab_size,
        "maxlen": pad_maxlen,
        "maxlen_max": pad_maxlen_max,
        "maxlen_min": pad_maxlen_min,
        "embedded_vectors": embedded_vectors,
        "tokenizer": tokenizer
    }

    return result


def tokenizer(data, tokenization_method_type, num_words=100000, pad_maxlen=0, embedded_path=""):
    embedded_vectors = []
    if tokenization_method_type == "simple":
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(data)
        train_data = tokenizer.texts_to_sequences(data)

        vocab_size = len(tokenizer.word_index) + 1

        if os.path.isfile(embedded_path):
            embedded_vectors = load_embedding_space(path=embedded_path,
                                                    vocab_size=vocab_size,
                                                    tokenizer=tokenizer)
    else:
        if "bert" in tokenization_method_type:
            tokenizer = BertTokenizer.from_pretrained(tokenization_method_type)
        elif "albert" in tokenization_method_type:  # 'albert-base-v1'
            tokenizer = AlbertTokenizer.from_pretrained()

        train_data = []
        with click.progressbar(data, label="TOKENIZATION: ", fill_char=click.style('=', fg='white')) as data_bar:
            for item in data_bar:
                tokenized_text = tokenizer.tokenize(item)
                indexed_tokens = tokenizer.convert_tokens_to_ids(
                    tokenized_text)
                train_data.append(indexed_tokens)

        vocab_size = tokenizer.vocab_size + 1

    pad_maxlen_lens = [len(item) for item in data]
    pad_maxlen_lens.sort()
    max_element_index = int(len(pad_maxlen_lens)*0.99)
    pad_maxlen_max = pad_maxlen_lens[max_element_index]
    pad_maxlen_min = min(pad_maxlen_lens)
    if pad_maxlen == 0:
        pad_maxlen = sum(pad_maxlen_lens) // len(data)

    feauters = pad_sequences(train_data,
                             padding='post',
                             maxlen=pad_maxlen)

    result = {
        "feauters": feauters,
        "vocab_size": vocab_size,
        "maxlen": pad_maxlen,
        "maxlen_max": pad_maxlen_max,
        "maxlen_min": pad_maxlen_min,
        "embedded_vectors": embedded_vectors,
        "tokenizer": tokenizer
    }

    return result
