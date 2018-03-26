import os
import sys
import argparse
import logging
import traceback
import subprocess
import math
import gzip
import re
import yaml
import sklearn
from sklearn.externals import joblib
import numpy as np

from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer

from mosestokenizer import MosesTokenizer

from features import feature_extract, Features
from prob_dict import ProbabilisticDictionary
from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup


#import cProfile  # search for "profile" throughout the file

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 28/12/2017 # Initial release # Sergio Ortiz"


# All the scripts should have an initialization according with the usage. Template:
def initialization():
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rt'), default=None, help="Tab-separated files to be classified")
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output of the classification")

    ## Parameters required
    groupM = parser.add_argument_group('Mandatory')
    groupM.add_argument('-m', '--metadata', type=argparse.FileType('r'), required=True, help="Training metadata (YAML file). Take into account that explicit command line arguments will overwrite the values from metadata file")

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument("-s", "--source_lang", type=str, help="Source language (SL) of the input")
    groupO.add_argument("-t", "--target_lang", type=str, help="Target language (TL) of the input")
    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")
    groupO.add_argument('--normalize_by_length', action='store_true', help="Normalize by length in qmax dict feature")
    groupO.add_argument('--treat_oovs', action='store_true', help="Special treatment for OOVs in qmax dict feature")
    groupO.add_argument('--qmax_limit', type=check_positive_or_zero, default=20, help="Number of max target words to be taken into account, sorted by length")    
    groupO.add_argument('--disable_features_quest', action='store_false', help="Disable less important features")
    groupO.add_argument('-g', '--good_examples',  type=check_positive_or_zero, default=50000, help="Number of good examples")
    groupO.add_argument('-w', '--wrong_examples', type=check_positive_or_zero, default=50000, help="Number of wrong examples")
    groupO.add_argument('--good_test_examples',  type=check_positive_or_zero, default=2000, help="Number of good test examples")
    groupO.add_argument('--wrong_test_examples', type=check_positive_or_zero, default=2000, help="Number of wrong test examples")
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--threshold', type=check_positive_between_zero_and_one, default=0.5, help="Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default.")
    
    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")

    # Validating & parsing
    # Checking if metadata is specified
    preliminary_args = parser.parse_args()
    if preliminary_args.metadata != None:
        # If so, we load values from metadata
        metadata_yaml = yaml.load(preliminary_args.metadata)
        threshold = np.argmax(metadata_yaml["accuracy_histogram"])*0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold
        logging.debug("YAML")
        logging.debug(metadata_yaml)
        parser.set_defaults(**metadata_yaml)
    # Then we build again the parameters to overwrite the metadata values if their options were explicitly specified in command line arguments
    args = parser.parse_args()
    logging_setup(args)
    
    # Extra-checks for args here
    # Load dictionaries
    args.dict_sl_tl = ProbabilisticDictionary(args.source_dictionary)
    args.dict_tl_sl = ProbabilisticDictionary(args.target_dictionary)
    # Load classifier
    args.clf = joblib.load(args.classifier)

    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

#def profile_classifier_process(i, jobs_queue, output_queue,args):
#    cProfile.runctx('classifier_process(i, jobs_queue, output_queue, args)', globals(), locals(), 'profiling-{}.out'.format(i))

def classifier_process(i, jobs_queue, output_queue, args):
    with MosesTokenizer(args.source_lang) as source_tokenizer, MosesTokenizer(args.target_lang) as target_tokenizer:
        while True:
            job = jobs_queue.get()
            if job:
                logging.debug("Job {0}".format(job.__repr__()))
                nblock, filein_name = job
                ojob = None
                with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir) as fileout:
                    logging.debug("Classification: creating temporary filename {0}".format(fileout.name))
                    feats = []

                    for i in filein:
                        parts = i.split("\t")
                        if len(parts) >= 2 and len(parts[0].strip()) != 0 and len(parts[1].strip()) != 0:
                            features = feature_extract(i, source_tokenizer, target_tokenizer, args)
                            # print("SENTENCE PAIR: %%{}%%".format(i))
                            # print(Features(features)) # debug
                            feats.append([float(v) for v in features])

                    predictions = args.clf.predict_proba(np.array(feats)) if len(feats) > 0 else []
                    filein.seek(0)

                    piter = iter(predictions)
                    for i in filein:
                        parts = i.split("\t")
                        if len(parts) >= 2 and len(parts[0].strip()) != 0 and len(parts[1].strip()) != 0:
                            p = next(piter)
                            fileout.write(i.strip())
                            fileout.write("\t")
                            fileout.write(str(p[1]))
                            fileout.write("\n")
                        else:
                            fileout.write(i.strip("\n"))
                            fileout.write("\t0\n")

                    ojob = (nblock, fileout.name)
                    filein.close()
                    fileout.close()
                 
                if ojob:                    
                    output_queue.put(ojob)
                    
                os.unlink(filein_name)
            else:
                logging.debug("Exiting worker")
                break

def mapping_process(args, jobs_queue):
    logging.info("Start mapping")
    nblock = 0
    nline = 0
    mytemp = None
    for line in args.input:
        if (nline % args.block_size) == 0:
            logging.debug("Creating block {}".format(nblock))
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        mytemp.write(line)
#        parts = line.strip().split("\t")
#        
#        if len(parts) == 2:
#            mytemp.write(line)
#        else:
#            logging.debug("Line not included in process: {}".format(line))
        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()        
        jobs_queue.put(job)

    return nline

def reduce_process(output_queue, args):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {0}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    parts = i.split("\t")
                    if len(parts) == 3:
                        pred = float(parts[2].strip())
                        args.output.write(i.strip("\n"))
                        if pred < args.threshold:
                            args.output.write("\tdiscard\n")
                        else:
                            args.output.write("\tkeep\n")

                        if args.discarded_tus:
                            args.discarded_tus.write(i)
                filein.close()
            os.unlink(filein_name)

        job = output_queue.get()
        if job:
            nblock, filein_name = job
            heappush(h, (nblock, filein_name))
        else:
            logging.debug("Exiting reduce loop")
            break

    if len(h) > 0:
        logging.debug("Still elements in heap")

    while len(h) > 0 and h[0][0] == last_block:
        nblock, filein_name = heapq.heappop(h)
        last_block += 1

        with open(filein_name, 'r') as filein:
            for i in filein:
                parts = i.split("\t")
                if len(parts) == 3:
                    pred = float(parts[2].strip())
                    args.output.write(i.strip("\n"))
                    if pred < args.threshold:
                        args.output.write("\tdiscard\n")
                    else:
                        args.output.write("\tkeep\n")

                    if args.discarded_tus:
                        args.discarded_tus.write(i)
            filein.close()

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Classification finished. Output available in {}".format(args.output.name))
    args.output.close()
    if args.discarded_tus:
        logging.info("Discarded TUs are available in {}".format(args.discarded_tus.name))
        args.discarded_tus.close()

# Filtering input texts
def perform_classification(args):
    time_start = default_timer()
    logging.info("Starting process")
    logging.info("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize = maxsize)
    worker_count = process_count

    # Start reducer
    reduce = Process(target = reduce_process,
                     args   = (output_queue, args))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize = maxsize)
    workers = []
    for i in range(worker_count):
        filter = Process(target = classifier_process, #profile_classifier_process
                         args   = (i, jobs_queue, output_queue, args))
        filter.daemon = True # dies with the parent process

        filter.start()
        workers.append(filter)

    # Mapper process (foreground - parent)
    nline = mapping_process(args, jobs_queue)
    args.input.close()

    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    logging.info("End mapping")

    for w in workers:
        w.join()

    # Reducer termination
    output_queue.put(None)
    reduce.join()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))
### END PARALLELIZATION METHODS ###

def main(args):
    logging.info("Executing main program...")
    perform_classification(args)
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
