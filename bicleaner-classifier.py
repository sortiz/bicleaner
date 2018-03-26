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
import xml.parsers.expat
import numpy as np

from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer

from mosestokenizer import MosesTokenizer

from features import feature_extract
from prob_dict import ProbabilisticDictionary
from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, check_if_folder

__author__ = "Jorge Ferrández-Tordera"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "Version 0.1 # 19/12/2017 # Initial version # Jorge Ferrández-Tordera"

# TODO use something more standard
def escape(str):
  return str.replace("\\","\\\\").replace("\n","\\n").replace("\t","\\t")

def unescape(str):
  return str.replace("\\n", "\n").replace("\\t","\t").replace("\\\\","\\")

def parse(args):
    logging.info("Parsing TMX file received as input {}".format(args.input.name))
    langpair = []
    tuid = -1
    output = args.temp_tmx_info

    def xml_decl(version, encoding, standalone):
        attrs = []
        if version:
            attrs.append('version="{}"'.format(version))
        if encoding:
            attrs.append('encoding="{}"'.format(encoding))
        if standalone and standalone != -1:
            attrs.append('standalone="{}"'.format("yes" if standalone == 1 else "no"))
        output.write(escape("<?xml {}?>\n".format(' '.join(attrs))))

    def start_element(name, attrs):
        nonlocal langpair
        nonlocal tuid
        elem = "<{0}{1}>".format(name, "".join([' {0}="{1}"'.format(i, attrs[i]) for i in attrs]))
        if name == "tu":
            if len(langpair) == 2:
                output.write("\t")
                output.write(langpair[0])
                output.write("\t")
                output.write(langpair[1])
                langpair = []
            output.write("\n")
            if "tuid" in attrs:
                tuid = attrs["tuid"]
        if name == "tuv":
            if "lang" in attrs:
                langpair.append(attrs["lang"])
            elif "xml:lang" in attrs:
                langpair.append(attrs["xml:lang"])
            else:
                raise Exception("The segments of TU {} are not identified with lang tag".format(tuid))

        output.write(escape(elem))

        if name == "seg":
            output.write("\t")

    def end_element(name):
        nonlocal langpair
        if name == "seg":
            output.write("\t")

        output.write("</{0}>".format(name))

        if name == "tmx":
            if len(langpair) == 2:
                output.write("\t")
                output.write(langpair[0])
                output.write("\t")
                output.write(langpair[1])

    def character_data(data):
        output.write(escape(data))

    parser = xml.parsers.expat.ParserCreate()
    parser.buffer_text = True
    parser.XmlDeclHandler = xml_decl
    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    parser.CharacterDataHandler = character_data
    parser.ParseFile(args.input)

    args.temp_tmx_info.seek(0)
    args.input.close()
    logging.info("Finished parsing of TMX file. Result in {}".format(output.name))

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rb'), default=None, help="TMX file with TUs to be classified")
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output of the TMX file with the classification integrated. New fields are added to the TMX received as input")

    ## Parameters required
    groupM = parser.add_argument_group('Mandatory')
    groupM.add_argument('-s', '--source_lang', required=True, type=str, help="Source language (SL) of the input")
    groupM.add_argument('-t', '--target_lang', required=True, type=str, help="Target language (TL) of the input")
    groupM.add_argument('-c', '--classifier', type=argparse.FileType('rb'), required=True, help="Classifier data file")
    groupM.add_argument('--source_dictionary',  type=argparse.FileType('r'), required=True, help="SL-TL gzipped probabilistic dictionary")
    groupM.add_argument('--target_dictionary', type=argparse.FileType('r'), required=True, help="TL-SL gzipped probabilistic dictionary")

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument('--tmp_dir', type=check_if_folder, default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('--remove_tmp', action='store_true', default=False, help="This flag specifies whether removing temporal files created by this program")
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")
    groupO.add_argument('-m', '--metadata', type=argparse.FileType('r'), help="Training metadata (YAML file). Take into account that explicit command line arguments will overwrite the values from metadata file")
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
        metadata_yaml["threshold"] = threshold
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

    # Temporary file for processed TMX
    args.temp_tmx_info = NamedTemporaryFile(mode="w+", delete=args.remove_tmp, dir=args.tmp_dir)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

#### PARALLELIZATION METHODS ###
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
                    # TODO Test times with predict one-by-one and this impl
                    for i in filein:
                        features = feature_extract(i, source_tokenizer, target_tokenizer, args)
                        feats.append([float(v) for v in features])

                    if len(feats) > 0:
                        prediction = args.clf.predict_proba(np.array(feats))

                        row = 0
                        for pred in prediction:
                            fileout.write("{}\n".format(str(pred[1])))
                            row += 1
                    
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
    for line in args.temp_tmx_info: # Reading the temp file with the TMX processed
        if (nline % args.block_size) == 0:
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        parts = line.strip().split("\t")
        if len(parts) == 7:
            # Last two columns are the language pair
            if parts[-2] == args.source_lang and parts[-1] == args.target_lang:
                mytemp.write("{}\t{}\n".format(parts[1], parts[3]))
            elif parts[-1] == args.source_lang and parts[-2] == args.source_lang:
                mytemp.write("{}\t{}\n".format(parts[3], parts[1]))
        else:
            logging.debug("Line not included in process: {}".format(line))
        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()
        jobs_queue.put(job)

    args.temp_tmx_info.seek(0)

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
                    must_break = False
                    for line in args.temp_tmx_info:
                        parts = line.strip().split("\t")
                        if parts[0].startswith("<tu "):
                            pred = float(i.strip())
                            # Check if it must be discarded
                            discarded = "no"
                            if pred < args.threshold:
                                discarded = "yes"
                                if args.discarded_tus:
                                    args.discarded_tus.write("{}\t{}\t{}\t{}\t{}\n".format(parts[1], parts[3], parts[-2], parts[-1], pred))
                            parts[0] = re.sub(r'>((\\n)*\s*)<', '>\g<1><prop type="filter-score">{}</prop>\g<1><prop type="discarded">{}</prop>\g<1><'.format(pred, discarded), parts[0], count=1)
                            must_break = True
                            for p in parts[:-2]:
                                args.output.write(unescape(p.strip("\n")))
                        else:
                            for p in parts:
                                args.output.write(unescape(p.strip("\n")))
                        if must_break:
                            break
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
                must_break = False
                for line in args.temp_tmx_info:
                    parts = line.strip().split("\t")
                    if parts[0].startswith("<tu "):
                        pred = float(i.strip())
                        # Check if it must be discarded
                        discarded = "no"
                        if pred < args.threshold:
                            discarded = "yes"
                            if args.discarded_tus:
                                args.discarded_tus.write("{}\t{}\t{}\t{}\t{}".format(parts[1], parts[3], parts[-2], parts[-1], pred))
                        parts[0] = re.sub(r'>((\\n)*\s*)<', '>\g<1><prop type="filter-score">{}</prop>\g<1><prop type="discarded">{}</prop>\g<1><'.format(pred, discarded), parts[0], count=1)
                        must_break = True
                        for p in parts[:-2]:
                            args.output.write(unescape(p.strip("\n")))
                    else:
                        for p in parts:
                            args.output.write(unescape(p.strip("\n")))
                    if must_break:
                        break
            filein.close()
        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Classification finished. Output TMX available in {}".format(args.output.name))
    args.temp_tmx_info.close()
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
        filter = Process(target = classifier_process,
                         args   = (i, jobs_queue, output_queue, args))
        filter.daemon = True # dies with the parent process
        filter.start()
        workers.append(filter)

    # Mapper process (foreground - parent)
    nline = mapping_process(args, jobs_queue)

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
    parse(args)
    perform_classification(args)
    logging.info("Program finished")

if __name__ == '__main__':
    try:
        args = initialization() # Parsing parameters
        main(args)  # Running main program
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
