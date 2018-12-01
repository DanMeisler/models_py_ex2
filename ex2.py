from collections import Counter
import argparse
import re

LABMDAS_VALUES = [round(x * 0.01, 2) for x in xrange(0, 201)]
SET_FILE_HEADER_LINE_REGEX = "<.*>"
VOCABULARY_SIZE = 300000
LIDSTONE_TRAINING_DEVELOPMENT_RATIO = 0.9


class OutputManager(object):
    STUDENTS = {"aaa": "111", "bbb": "222"}

    def __init__(self, output_file_handle):
        self._output_file_handle = output_file_handle
        self._output_count = 1
        self._output_students_line()

    def output(self, data):
        self._output_file_handle.write("#Output%d\t%s\n" % (self._output_count, data))
        self._output_count += 1

    def _output_students_line(self):
        students_line_parts = ["#Studnets"]
        students_line_parts.extend(self.STUDENTS.keys())
        students_line_parts.extend(self.STUDENTS.values())
        self._output_file_handle.write("\t".join(students_line_parts) + "\n")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('development_set_file_path', help='The path to the development set file')
    parser.add_argument('test_set_file_path', help='The path to the test set file')
    parser.add_argument('input_word', help='The word the analyse')
    parser.add_argument('output_file_path', help='The path to the output file')
    return parser.parse_args()


def get_set_from_file(set_file_path):
    a_set = []
    with open(set_file_path, "r") as set_file_handle:
        for line in set_file_handle:
            if not re.match(SET_FILE_HEADER_LINE_REGEX, line):
                a_set.extend(line.strip().split())
    return a_set


def main(args):
    output_file_handle = open(args.output_file_path, "w")
    output_manager = OutputManager(output_file_handle)
    # 1.Init
    output_manager.output(args.development_set_file_path)
    output_manager.output(args.test_set_file_path)
    output_manager.output(args.input_word)
    output_manager.output(args.output_file_path)
    output_manager.output(VOCABULARY_SIZE)
    output_manager.output(1.0 / VOCABULARY_SIZE)
    # 2. Development set preprocessing
    development_set = get_set_from_file(args.development_set_file_path)
    output_manager.output(len(development_set))
    # 3. Lidstone model training
    lidstone_split_index = int(round(LIDSTONE_TRAINING_DEVELOPMENT_RATIO * len(development_set)))
    training_set, validation_set = development_set[:lidstone_split_index], development_set[lidstone_split_index:]
    output_manager.output(len(validation_set))
    output_manager.output(len(training_set))
    training_set_counter = Counter(training_set)
    output_manager.output(len(training_set_counter))
    output_manager.output(training_set_counter[args.input_word])

    # Done
    output_file_handle.close()


if __name__ == "__main__":
    main(get_arguments())
