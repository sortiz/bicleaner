import os
import sys
import argparse
import logging
import traceback
import re
import subprocess
import math
import gzip

from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer

from util import no_escaping, logging_setup, check_if_folder

from mosestokenizer import * # Dependency https://pypi.python.org/pypi/mosestokenizer/ - sudo pip install mosestokenizer

__author__ = "Jorge Ferrández-Tordera"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "Version 0.1 # 11/12/2017 # Initial version # Jorge Ferrández-Tordera"

# Moses decoder scripts used - mosesdecoder folder should exist
TRAIN_MODEL_SCRIPT = "train-model.perl"
GIZA_MKCLS="mkcls"
GIZA_MGIZA="mgiza"
GIZA_SNT2COOC="snt2cooc"

# Names for ouput files
CLEAN_OUTPUT = "corpus-cleaned"
DICT_SL_TL = "model/lex.f2e"
DICT_SL_TL_SORTED = DICT_SL_TL + ".sorted"
DICT_TL_SL = "model/lex.e2f"
DICT_TL_SL_SORTED = DICT_TL_SL + ".sorted"
DICT_FINAL_NAME = "dict.{}-{}.gz"

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('r'), default=sys.stdin,  help="Tab-separated bilingual input file")
    parser.add_argument('-o', '--output_dir', required=True, type=str, default=os.getcwd(), help="Output directory. Cleaned corpus and dictionary will be created here. Folder will be created if not exists")
    parser.add_argument('--giza', required=True, type=str, help="GIZA++ folder path, which contains binaries. Expected scripts in the folder: {}, {} and {}".format(GIZA_MKCLS, GIZA_MGIZA, GIZA_SNT2COOC))
    parser.add_argument('--moses_dir', required=True, type=str, help="Moses scripts folder path, which contains the script {}".format(TRAIN_MODEL_SCRIPT))

    ## Parameters required
    groupM = parser.add_argument_group('Mandatory')
    groupM.add_argument('-s', '--source_lang', required=True, type=str, help="Source language of the input")
    groupM.add_argument('-t', '--target_lang', required=True, type=str, help="Target language of the input")

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument('-m', '--tmp_dir', type=check_if_folder, default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")
    groupO.add_argument('-r', '--giza_ratio', type=float, default=9, help="9-1 Sentence ratio limit of GIZA++ (it shouldn't be modified)")
    groupO.add_argument('-n', '--prune_ratio', type=float, default=10, help="Ratio to prune the dictionary. Translations whose probability is {} times (default) than the maximum one.".format(10))
    groupO.add_argument('--min', type=int, default=1, help="Minimum number of tokens allowed for a sentence")
    groupO.add_argument('--max', type=int, default=50, help="Maximum number of tokens allowed for a sentence")

    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")

    # Validating & parsing
    args = parser.parse_args()
    logging_setup(args)

    # Extra-checks for args here
    args.prune_ratio = math.log(args.prune_ratio)
    if not args.output_dir.endswith("/"):
        args.output_dir = args.output_dir + "/"
    if not args.moses_dir.endswith("/"):
        args.moses_dir = args.moses_dir + "/"
    args.output_moses_corpus = args.output_dir + CLEAN_OUTPUT
    args.moses_train_script = args.moses_dir + TRAIN_MODEL_SCRIPT

    # Checking if moses scripts exist before running previous processes
    if not os.path.isdir(os.path.expanduser(args.output_dir)):
        logging.info("The output folder {} doesn't exist. Creating...".format(args.output_dir))
        os.makedirs(os.path.expanduser(args.output_dir))
        logging.info("Output folder created at {}".format(args.output_dir))
    if not os.path.isfile(args.moses_train_script):
        raise argparse.ArgumentTypeError("Moses script {} cannot be found in path {}".format(TRAIN_MODEL_SCRIPT, args.moses_train_script))
    if not os.path.isfile(args.giza + GIZA_MGIZA) or not os.path.isfile(args.giza + GIZA_MKCLS) or not os.path.isfile(args.giza + GIZA_SNT2COOC):
        raise argparse.ArgumentTypeError("Necessary GIZA++ scripts cannot be found in path {}. Please check if some of the following scripts are missing in the folder: {}, {} and {}".format(args.giza, GIZA_MGIZA, GIZA_MKCLS, GIZA_SNT2COOC))
    # Intermediary files
    args.output_source = open("{}{}".format(args.output_dir, CLEAN_OUTPUT + "." + args.source_lang), "w")
    args.output_target = open("{}{}".format(args.output_dir, CLEAN_OUTPUT + "." + args.target_lang), "w")

    # Final dicts names
    args.dict_sl_tl_final = "{}{}".format(args.output_dir, DICT_FINAL_NAME.format(args.source_lang, args.target_lang))
    args.dict_tl_sl_final = "{}{}".format(args.output_dir, DICT_FINAL_NAME.format(args.target_lang, args.source_lang))

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

# Validate the sentence in both languages
def validate_sentences(args, source_sentence, target_sentence):
    len_source = len(source_sentence)
    len_target = len(target_sentence)
    # Filtering by length
    if (len_source < args.min) or (len_target < args.min) or (len_source > args.max) or (len_target > args.max):
        logging.debug("Too long or too short sentence [{}]-[{}]: {}\t{}".format(len_source, len_target, source_sentence, target_sentence))
        return False
    # Filtering by 9-1 ratio limit of GIZA
    if (len_source / float(len_target) > args.giza_ratio) or (len_target / float(len_source) > args.giza_ratio):
        logging.debug("9-1 ratio limit excedeed: {}\t{}".format(source_sentence, target_sentence))
        return False
    return True

# Write to the output the sentences in both langauges separated by a tab
def write_sentences(args, source_sentence, target_sentence, output):
    # Undo XML escape of characters done by Moses before writing sentences
    output.write(no_escaping(source_sentence))
    output.write("\t")
    output.write(no_escaping(target_sentence))
    output.write("\n")

# Tokenize, split and filter every line of text
def filter_text(line, args, source_tokenizer, target_tokenizer, source_splitter, target_splitter, output):
    # Text is lowercased and split in source and target parts
    parts = line.strip().lower().split("\t")
    # If parts are empty, don't write them in the output
    if (len(parts) < 2) or bool(parts[0].strip()) == False or bool(parts[1].strip()) == False:
        logging.debug("Empty line or, at least, part of it: {}".format(line))
        return
    if len(parts) > 2:
        logging.debug("Incorrect format. Found more than one tab character in line: {}".format(line))
        return

    # Removing spaces
    multiple_spaces_re = re.compile('\s+')
    parts[0] = multiple_spaces_re.sub(' ', parts[0])
    parts[1] = multiple_spaces_re.sub(' ', parts[1])

    # Tokenizing both sentences
    source_sentence = source_tokenizer(parts[0])
    target_sentence = target_tokenizer(parts[1])

    # Splitting both sentences
    source_sentences_list = source_splitter(source_sentence)
    target_sentences_list = target_splitter(target_sentence)

    # Checking splitter output
    # If splitter returns same number of sentences in both languages
    if (len(source_sentences_list) == len(target_sentences_list)):
        for n, src_sentence in enumerate(source_sentences_list):
            trg_sentence = target_sentences_list[n]
            if validate_sentences(args, source_tokenizer(src_sentence), target_tokenizer(trg_sentence)):
                write_sentences(args, src_sentence, trg_sentence, output)
    else:
        src_sentence = ' '.join(source_sentence)
        trg_sentence = ' '.join(target_sentence)
        if validate_sentences(args, source_sentence, target_sentence):
            write_sentences(args, src_sentence, trg_sentence, output)

# Prune translations
def prune_translations(ratio, source, translations, output):
    max_prob = -sys.float_info.max

    # For each translation, probability and log(probability)
    for t, p, l in translations:
        if l > max_prob:
            max_prob = l
    threshold = max_prob - ratio
    for t, p, l in translations:
        if l >= threshold:
            output.write(str(source + " " + t + " " + p + "\n").encode("utf-8"))
        else:
            logging.debug("Translation {} discarded. Prob: {} - Max prob: {}".format(t, p, math.exp(threshold)))

# Prune bilingual probabilistic dictionary
# The method keeps only translations whose probability is 10 times (by default) lower than the maximum one
def prune_dictionary(dictionary, args, output):
    current_source = None
    current_translations = []
    for line in dictionary:
        parts = line.strip().split(" ")
        if current_source != None and parts[1] != current_source:
            prune_translations(args.prune_ratio, current_source, current_translations, output)
            current_translations = []
        current_source = parts[1]
        current_translations.append((parts[0], parts[2], math.log(float(parts[2]))))
    if len(current_translations) > 0:
        prune_translations(args.prune_ratio, current_source, current_translations, output)

# Align with GIZA++
def build_dictionaries(args):
    logging.info("Building dictionaries...")
    logging.info("Running Moses script ({}) for extracting bilingual dictionary in {}".format(args.moses_train_script, args.output_dir))
    train_model = ["perl", args.moses_train_script, "--parallel", "--mgiza", "--mgiza-cpus", str(args.processes),
                    "--external-bin-dir", args.giza, "--root-dir", args.output_dir, "--corpus", args.output_moses_corpus,
                    "--f", args.source_lang, "--e", args.target_lang, "--first-step", "1", "--last-step", "4",
                    "--sort-parallel", str(args.processes), "--temp-dir", args.tmp_dir]

    logging.debug("Moses command launched: {}".format(str(' '.join(train_model))))
    moses_process = subprocess.run(train_model, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("Moses script finished.")
    sort_dictionaries = ["sort", "-k2", "--parallel", str(args.processes), "-T", args.tmp_dir]
    logging.debug("'sort' commands will be launched as: {}".format(str(' '.join(sort_dictionaries))))
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    output_dict_sl=args.output_dir + DICT_SL_TL_SORTED
    output_dict_tl=args.output_dir + DICT_TL_SL_SORTED
    with open(args.output_dir + DICT_SL_TL, 'r') as input_sl, open(output_dict_sl, 'w+') as pruned_sl, gzip.open(args.dict_sl_tl_final, "wb") as output_sl:
        logging.debug("Dictionary {}-{} ({}): Sorting it in {} and pruning it in {}...".format(args.source_lang, args.target_lang, input_sl.name, pruned_sl.name, output_sl.name))
        sort_sl_process = subprocess.run(sort_dictionaries, stdin=input_sl, stdout=pruned_sl, env=env)
        pruned_sl.flush()
        pruned_sl.seek(0)
        prune_dictionary(pruned_sl, args, output_sl)
        logging.debug("Dictionary {}-{} ({}) pruned and compressed.".format(args.source_lang, args.target_lang, output_sl.name))
        input_sl.close()
        pruned_sl.close()
        output_sl.close()
    with open(args.output_dir + DICT_TL_SL, 'r') as input_tl, open(output_dict_tl, 'w+') as pruned_tl, gzip.open(args.dict_tl_sl_final, "wb") as output_tl:
        logging.debug("Dictionary {}-{} ({}): Sorting it in {} and pruning it in {}...".format(args.target_lang, args.source_lang, input_tl.name, pruned_tl.name, output_tl.name))
        sort_tl_process = subprocess.run(sort_dictionaries, stdin=input_tl, stdout=pruned_tl, env=env)
        pruned_tl.flush()
        pruned_tl.seek(0)
        prune_dictionary(pruned_tl, args, output_tl)
        logging.debug("Dictionary {}-{} ({}) pruned and compressed.".format(args.target_lang, args.source_lang, output_tl.name))
        input_tl.close()
        pruned_tl.close()
        output_tl.close()
    logging.info("Dictionaries are built, pruned and compressed in {} and {}".format(args.dict_sl_tl_final, args.dict_tl_sl_final))

#### PARALLELIZATION METHODS ###
def filter_process(i, jobs_queue, output_queue, args):
  with MosesTokenizer(args.source_lang) as source_tokenizer, MosesSentenceSplitter(args.source_lang) as source_splitter, \
      MosesTokenizer(args.target_lang) as target_tokenizer, MosesSentenceSplitter(args.target_lang) as target_splitter:
    while True:
        job = jobs_queue.get()
        if job:
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir) as fileout:
                logging.debug("Filtering: creating temporary filename {0}".format(fileout.name))
                for i in filein:
                    filter_text(i, args, source_tokenizer, target_tokenizer, source_splitter, target_splitter, fileout)                
                ojob = (nblock, fileout.name)
                fileout.close()
                filein.close()
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
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        mytemp.write(line)
        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()
        jobs_queue.put(job)

    args.input.close()

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
            parts = i.strip().split("\t")
            args.output_source.write(parts[0] + "\n")
            args.output_target.write(parts[1] + "\n")
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
          parts = i.strip().split("\t")
          args.output_source.write(parts[0] + "\n")
          args.output_target.write(parts[1] + "\n")
      filein.close()
      
    os.unlink(filein_name)

  if len(h) != 0:
    logging.error("The queue is not empty and it should!")

  args.output_source.close()
  args.output_target.close()

# Filtering input texts
def perform_filtering(args):
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
        filter = Process(target = filter_process,
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
    perform_filtering(args)
    build_dictionaries(args)
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
