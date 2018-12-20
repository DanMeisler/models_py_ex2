# GF *********
# DM *********

from collections import Counter, defaultdict
import argparse
import math
import re

LABMDAS_VALUES = [round(x * 0.01, 2) for x in range(1, 201)]
LIDSTONE_TRAINING_DEVELOPMENT_RATIO = 0.9
SET_FILE_HEADER_LINE_REGEX = "^<.*>$"
VOCABULARY_SIZE = 300000
UNSEEN_WORD = "unseen-word"


class OutputManager(object):
    STUDENTS = {"GF": "*********", "DM": "*********"}

    def __init__(self, output_file_path):
        self._output_file = open(output_file_path, "w")
        self._output_count = 1
        self._output_students_line()

    def __del__(self):
        self._output_file.close()

    def output(self, data):
        self._output_file.write("#Output%d\t%s\n" % (self._output_count, data))
        self._output_count += 1

    def _output_students_line(self):
        students_line_parts = ["#Studnets"]
        students_line_parts.extend(self.STUDENTS.keys())
        students_line_parts.extend(self.STUDENTS.values())
        self._output_file.write("\t".join(students_line_parts) + "\n")


class Estimators(object):
    def __init__(self):
        pass


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('development_set_file_path', help='The path to the development set file')
    parser.add_argument('test_set_file_path', help='The path to the test set file')
    parser.add_argument('input_word', help='The word to analyse')
    parser.add_argument('output_file_path', help='The path to the output file')
    return parser.parse_args()


def get_article_set(article_set_file_path):
    """
    :param article_set_file_path: the dataset's path
    :return: dataset as list
    """
    article_set = []
    with open(article_set_file_path, "r") as article_set_file:
        for line in article_set_file:
            line = line.strip()
            if not re.match(SET_FILE_HEADER_LINE_REGEX, line):
                article_set.extend(line.split())
    return article_set


def p_MLE(training_set_counter, training_set):
    """
    :param training_set_counter: Dictionary containing the frequency for each word
    :param training_set: training dataset
    :return: MLE function receiving word and returning its MLE probability (1 / |training_set|)
    """
    return lambda word: training_set_counter[word] / float(len(training_set)) if word in training_set_counter else 0.0


def lidstone_model(_lambda, training_counter, training_set_size):
    """
    :param _lambda: the lambda parameter
    :param training_counter: Dictionary containing the frequency for each word
    :param training_set_size: size of training set
    :return: a function that find p_lidstone for every word
    """
    return lambda word: float(training_counter[word] + _lambda) / (training_set_size + _lambda * VOCABULARY_SIZE)


def calc_perplexity(p_func, sample):
    """
    :param p_func: probability function (Lidstone, Heldout, MLE)
    :param sample: a list contains a dataset of words
    :return: the perplexity measure for this parameters
    """
    # Sum logs of words probabilities
    log_sum = sum([math.log(p_func(word), 2) for word in sample])

    # Divide -log_sum by len of dataset and use as exponent of 2 to receive perplexity
    exponent = -1 * log_sum / len(sample)
    return 2 ** exponent


def find_perplexity(_lambda, training_counter, training_set_size, validation_set):
    """
    :param _lambda: Lidstone lambda param
    :param training_counter: Dictionary containing the frequency for each word in training set
    :param training_set_size: size of training set
    :param validation_set: the part of the training dataset used as validation set
    :return: Lidstone's perplexity
    """

    # Calc Lidstone func using the received parameters
    p_lid = lidstone_model(_lambda, training_counter, training_set_size)

    # Return perplexity of validation_set
    return calc_perplexity(p_lid, validation_set)


def construct_lambda_dict(training_counter, training_set_size, validation_set):
    """
    This function construct dictionary of lambdas & perplexities according the Lidstone Model
    :param training_counter: Dictionary containing the frequency for each word in training set
    :param training_set_size: size of training set
    :param validation_set: the part of the training dataset used as validation set
    :return: lambda to perplexity dictionary
    """
    lambda_dictionary = {}
    for _lambda in LABMDAS_VALUES:
        lambda_dictionary[_lambda] = find_perplexity(_lambda, training_counter, training_set_size, validation_set)
    return lambda_dictionary


def construct_nr_dict(training_counter):
    """
    This function construct Dictionary of word frequency(count) to a list of all words in dataset
                                                                   that appears in this frequency
    :param training_counter: Dictionary containing the frequency for each word in training set
    :return: Dictionary from frequency to list of words matching this frequency
    """
    nr_dictionary = defaultdict(list)
    for k, v in training_counter.items():
        nr_dictionary[v].append(k)
    return nr_dictionary


def get_heldout_nr(r, train_counter):
    """
    :param r: r parameter of Heldout Model - meaning a word's frequency in training set
    :param train_counter: Dictionary containing the frequency for each word in training set
    :return: Nr parameter of Heldout Model - meaning number of words with r frequency in training set
    """

    # If r is 0 - number of words in vocabulary & not in training
    if not r:
        return VOCABULARY_SIZE - len(train_counter)
    else:
        return len(r)


def get_heldout_tr(r, train_counter, heldout_counter):
    """
    :param r: r parameter of Heldout Model - meaning a word's frequency in training set
    :param train_counter: Dictionary containing the frequency for each word in training set
    :param heldout_counter: Dictionary containing the frequency for each word in heldout set
    :return: tr parameter of Heldout Model - the total number of appearances on heldout set
                                                    of words with r frequency on training set
    """
    # If r is 0, sum all appearances on heldout set of words that doesn't appear on training set
    if not r:
        tr = sum([v for k, v in heldout_counter.items() if k not in train_counter])
    # Else, sum all words appearances on heldout set with r appearances on training set
    else:
        tr = sum([heldout_counter[word] for word in r])
    return tr


def heldout_model_for_r(r, train_counter, heldout_counter, heldout_set_size):
    """
    :param r: r parameter of Heldout Model - meaning a word's frequency in training set
    :param train_counter: Dictionary containing the frequency for each word in training set
    :param heldout_counter: Dictionary containing the frequency for each word in heldout set
    :param heldout_set_size: Heldout set size
    :return: p_heldout for this r
    """
    nr = get_heldout_nr(r, train_counter)
    tr = get_heldout_tr(r, train_counter, heldout_counter)

    # According Heldout Model
    return tr/(float(nr) * float(heldout_set_size))


def main(args):

    # Output class for the exercise
    output_manager = OutputManager(args.output_file_path)

    # 1. Init

    # Output 1
    output_manager.output(args.development_set_file_path)

    # Output 2
    output_manager.output(args.test_set_file_path)

    # Output 3
    output_manager.output(args.input_word)

    # Output 4
    output_manager.output(args.output_file_path)

    # Output 5
    output_manager.output(VOCABULARY_SIZE)

    # Output 6
    output_manager.output(1.0 / VOCABULARY_SIZE)

    # 2. Development set preprocessing

    # Output 7
    development_set = get_article_set(args.development_set_file_path)
    output_manager.output(len(development_set))

    # 3. Lidstone model training

    # Output 8
    lidstone_split_index = int(round(LIDSTONE_TRAINING_DEVELOPMENT_RATIO * len(development_set)))
    training_set, validation_set = development_set[:lidstone_split_index], development_set[lidstone_split_index:]
    output_manager.output(len(validation_set))

    # Output 9
    output_manager.output(len(training_set))

    # Output 10

    # Dictionary containing the frequency for each different word in training set
    training_set_counter = Counter(training_set)
    output_manager.output(len(training_set_counter))

    # Output 11
    output_manager.output(training_set_counter[args.input_word])

    # Output 12
    p_mle = p_MLE(training_set_counter, training_set)
    output_manager.output(p_mle(args.input_word))

    # Output 13
    output_manager.output(p_mle(UNSEEN_WORD))

    # Output 14
    p_lid_10 = lidstone_model(0.10, training_set_counter, len(training_set))
    output_manager.output(p_lid_10(args.input_word))

    # Output 15
    output_manager.output(p_lid_10(UNSEEN_WORD))

    # Output 16
    lambda_to_perplexity = construct_lambda_dict(training_set_counter, len(training_set), validation_set)
    output_manager.output(lambda_to_perplexity[0.01])

    # Output 17
    output_manager.output(lambda_to_perplexity[0.10])

    # Output 18
    output_manager.output(lambda_to_perplexity[1.00])

    # Output 19

    # Argmin of lambda_to_perplexity - lambda with minimal perplexity
    best_lambda = min(lambda_to_perplexity, key=lambda_to_perplexity.get)
    output_manager.output(best_lambda)

    # Output 20
    output_manager.output(lambda_to_perplexity[best_lambda])

    # Save Counter for Debug part
    lidstone_train_counter = training_set_counter
    best_lidstone = lidstone_model(best_lambda, lidstone_train_counter, len(training_set))

    # 4. Held Out Model Training

    # Output 21

    # Split in half
    heldout_split_index = int(round(0.5 * len(development_set)))
    training_set, heldout_set = development_set[:heldout_split_index], development_set[heldout_split_index:]
    output_manager.output(len(training_set))

    # Output 22
    output_manager.output(len(heldout_set))

    # Output 23

    # Define Counters for training and heldout sets
    training_set_counter = Counter(training_set)
    heldout_set_counter = Counter(heldout_set)

    # Construct Nr dictionary
    nr_dictionary = construct_nr_dict(training_set_counter)

    # p_heldout is function receiving word and returning its heldout
    p_heldout = lambda word: heldout_model_for_r(nr_dictionary[training_set_counter[word]], training_set_counter, heldout_set_counter, len(heldout_set))
    output_manager.output(p_heldout(args.input_word))

    # Output 24
    output_manager.output(p_heldout(UNSEEN_WORD))

    # 5. Debug

    # Lidstone

    # Number of words on vocabulary that does not exist in lidstone_train
    lidstone_unseen_words_count = VOCABULARY_SIZE - len(lidstone_train_counter)

    # Total probabilities - number of unseen words * p_lid(UNSEEN_WORD)
    lidstone_unseen_total = lidstone_unseen_words_count * best_lidstone(UNSEEN_WORD)

    # Sum of all probabilities for words on the lidstone train
    lidstone_seen_total = sum(best_lidstone(word) for word in lidstone_train_counter)

    # Verify total sum is 1
    assert round(lidstone_seen_total + lidstone_unseen_total, 8) == 1.0, "Lidstone Model Error!"

    # Heldout

    # Number of words on vocabulary that does not exist in heldout_train
    heldout_unseen_words_count = VOCABULARY_SIZE - len(training_set_counter)

    # Total probabilities - number of unseen words * p_heldout(UNSEEN_WORD)
    heldout_unseen_total = heldout_unseen_words_count * p_heldout(UNSEEN_WORD)

    # Sum of all probabilities for words on the heldout train
    heldout_seen_total = sum(p_heldout(word) for word in training_set_counter)

    # Verify total sum is 1
    assert round(heldout_seen_total + heldout_unseen_total, 8) == 1.0, "Heldout Model Error!"

    # 6. Models evaluation on test set

    # Output 25
    test_set = get_article_set(args.test_set_file_path)
    output_manager.output(len(test_set))

    # Output 26
    lidstone_test_perplexity = calc_perplexity(best_lidstone, test_set)
    output_manager.output(lidstone_test_perplexity)

    # Output 27
    heldout_test_perplexity = calc_perplexity(p_heldout, test_set)
    output_manager.output(heldout_test_perplexity)

    # Output 28
    output_manager.output('H' if heldout_test_perplexity < lidstone_test_perplexity else 'L')

    # Output 29
    parameters_table = ''
    for i in range(10):
        r = nr_dictionary.get(i)
        nr = get_heldout_nr(r, training_set_counter)
        tr = get_heldout_tr(r, training_set_counter, heldout_set_counter)
        fh = heldout_model_for_r(r, training_set_counter, heldout_set_counter, len(heldout_set)) * len(training_set)
        fg = (best_lambda + float(i)) * lidstone_split_index / (lidstone_split_index + best_lambda * VOCABULARY_SIZE)
        parameters_table += '\n' + '\t'.join([str(i), str(round(fg, 5)), str(round(fh, 5)), str(nr), str(tr)])
    output_manager.output(parameters_table)


if __name__ == "__main__":
    main(get_arguments())
