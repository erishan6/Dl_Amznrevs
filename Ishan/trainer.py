import sys
from copy import deepcopy


def get_corpus(filename):
    f = open(filename)
    corpus = []
    sentence_pair = {}
    for line in f.readlines():
        if line[0].upper() == 'E':
            # this is an english sentence
            sentence_pair["english"] = line.strip()
        if line[0].upper() == 'F':
            # this is an foreign sentence
            sentence_pair["foreign"] = line.strip()
            corpus.append(sentence_pair)
            sentence_pair = {}
    return corpus


def get_unique_words(corpus):
    english_words = set()
    foreign_words = set()
    for sentence_pair in corpus:
        for word in sentence_pair["english"].split():
            english_words.add(word)
        for word in sentence_pair["foreign"].split():
            foreign_words.add(word)

    return {"english": english_words, "foreign": foreign_words}


def init_probabilities(corpus):
    words = get_unique_words(corpus)
    probabilities = {}
    for word_english in words['english']:
        probabilities[word_english] = {}
        for word_foreign in words["foreign"]:
            probabilities[word_english][word_foreign] = 1 / len(words['english'])
    return probabilities


def perform_iteration(corpus, words, total_s, previous_probabilities):
    probabilities = deepcopy(previous_probabilities)

    count = {}
    for word_english in words['english']:
        count[word_english] = {}
        for word_foreign in words["foreign"]:
            count[word_english][word_foreign] = 0

    total = {}
    for word_foreign in words['foreign']:
        total[word_foreign] = 0

    for sentence_pair in corpus:
        e_s = sentence_pair["english"]
        f_s = sentence_pair["foreign"]
        for e in e_s.split():
            total_s[e] = 0

            for f in f_s.split():
                total_s[e] += probabilities[e][f]

        for e in e_s.split():
            for f in f_s.split():
                count[e][f] += (probabilities[e][f] / total_s[e])
                total[f] += probabilities[e][f] / total_s[e]

    for f in words['foreign']:
        for e in words['english']:
            probabilities[e][f] = count[e][f] / total[f]

    return probabilities


def perform_em_algo(corpus):
    words = get_unique_words(corpus)
    total_s = {word_english: 0 for word_english in words['english']}
    previous_probabilities = init_probabilities(corpus)

    iterations = 0
    while iterations < 100:
        new_probabilities = perform_iteration(corpus, words, total_s, previous_probabilities)

        previous_probabilities = new_probabilities
        iterations += 1
    return new_probabilities


if __name__ == '__main__':
    corpus = get_corpus(sys.argv[1])

    probabilities = perform_em_algo(corpus)
    print(probabilities)
