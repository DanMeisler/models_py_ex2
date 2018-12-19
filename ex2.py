from collections import Counter
import argparse
import math
import re

LABMDAS_VALUES = [round(x * 0.01, 2) for x in range(1, 201)]
LIDSTONE_TRAINING_DEVELOPMENT_RATIO = 0.9
SET_FILE_HEADER_LINE_REGEX = "^<.*>$"
VOCABULARY_SIZE = 300000
UNSEEN_WORD = "unseen-word"


class OutputManager(object):
    STUDENTS = {"aaa": "111", "bbb": "222"}

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
    article_set = []
    with open(article_set_file_path, "r") as article_set_file:
        for line in article_set_file:
            line = line.strip()
            if not re.match(SET_FILE_HEADER_LINE_REGEX, line):
                article_set.extend(line.split())
    return article_set


def lidstone_model(_lambda, training_counter, training_set_size):
    return lambda word: float(training_counter[word] + _lambda) / (training_set_size + _lambda * VOCABULARY_SIZE)


def calc_perplexity(p_func, sample):
    log_sum = sum([math.log(p_func(word), 2) for word in sample])
    exponent = -1 * log_sum / len(sample)
    return 2 ** exponent


def find_perplexity(_lambda, training_counter, training_set_size, validation_set):
    p_lid = lidstone_model(_lambda, training_counter, training_set_size)
    return calc_perplexity(p_lid, validation_set)


def construct_lambda_dict(training_counter, training_set_size, validation_set):
    lambda_dictionary = {}
    for _lambda in LABMDAS_VALUES:
        lambda_dictionary[_lambda] = find_perplexity(_lambda, training_counter, training_set_size, validation_set)
    return lambda_dictionary


def main(args):
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
    training_set_counter = Counter(training_set)
    output_manager.output(len(training_set_counter))

    # Output 11
    output_manager.output(training_set_counter[args.input_word])

    # Output 12
    p_mle = lambda word: training_set_counter[word]/float(len(training_set)) if word in training_set_counter else 0.0
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
    best_lambda = min(lambda_to_perplexity, key=lambda_to_perplexity.get)
    output_manager.output(best_lambda)

    # Output 20
    output_manager.output(lambda_to_perplexity[best_lambda])


if __name__ == "__main__":
    main(get_arguments(),)
