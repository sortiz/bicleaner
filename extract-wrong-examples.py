import os
import sys
import argparse
import logging
import traceback
import subprocess
import math
import gzip
import re
import random

from tempfile import NamedTemporaryFile, gettempdir

from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, check_if_folder

__author__ = "Jorge Ferrández-Tordera"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "Version 0.1 # 09/01/2018 # Initial version. Extract wrong_examples from the corpus classified  # Jorge Ferrández-Tordera"

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('r'), default=sys.stdin, help="TSV previously classified to extract bad examples")
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output with the bad examples selected in the process")

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument('--tmp_dir', type=check_if_folder, default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-w', '--wrong_examples', type=check_positive_or_zero, default=50000, help="Number of wrong examples")
    groupO.add_argument('--wrong_test_examples', type=check_positive_or_zero, default=2000, help="Number of wrong test examples")
    groupO.add_argument('--threshold', type=check_positive_between_zero_and_one, default=None, help="Threshold for classifier.")

    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")

    # Validating & parsing
    args = parser.parse_args()
    logging_setup(args)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

def extract_bad_examples(args):
    logging.info("Shuffle starts")
    total_size   = 0

    with NamedTemporaryFile("w+", dir=args.tmp_dir) as temp:
        logging.info("Indexing file")
        # (1) Calculate the number of lines, offsets
        offsets = []
        nline = 0
        ssource = 0
        starget = 0
        count = 0

        for line in args.input:
            total_size += 1
            parts = line.strip().split("\t")
            if args.threshold != None:
                if float(parts[2]) < args.threshold:
                    offsets.append(count)
                    count += len(bytearray(line, "UTF-8"))
                    nline += 1
                    temp.write(line)     
               
            elif parts[3] == "discard":
                offsets.append(count)
                count += len(bytearray(line, "UTF-8"))
                nline += 1
                temp.write(line)


        logging.info("Shuffling wrong sentences")
        # (2) Get wrong sentences
        random.shuffle(offsets)

        min_range = min(nline, args.wrong_examples + args.wrong_test_examples)
        for i in offsets[0:min_range]:
            temp.seek(i)
            args.output.write(temp.readline())

    logging.info("End shuffle. Output file: {}".format(args.output.name))
    return total_size

def main(args):
    logging.info("Extracting bad examples from last classification received {}".format(args.input.name))
    extract_bad_examples(args)

    args.output.close()
    args.input.close()
    logging.info("Program finished")

if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization() # Parsing parameters
        main(args)  # Running main program
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
