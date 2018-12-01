import argparse

VOCABULARY_SIZE = 300000
LABMDAS_VALUES = [round(x * 0.01, 2) for x in xrange(0, 201)]


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


def main(args):
    output_file_handle = open(args.output_file_path, "w")
    output_manager = OutputManager(output_file_handle)
    output_manager.output(args.development_set_file_path)
    output_file_handle.close()


if __name__ == "__main__":
    main(get_arguments())
