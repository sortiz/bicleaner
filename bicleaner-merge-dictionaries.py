#!/usr/bin/python
# -*- coding: utf8 -*-
import os
import re
import sys
import argparse
import logging
import traceback
import tempfile
import subprocess
import operator
import math
import gzip
import shutil

from util import check_if_folder
from util import logging_setup
from tempfile import gettempdir

__author__ = "mbanon"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "0.1 # 20180104 # Probabilistic dictionary merger # mbanon"

# All the scripts should have an initialization according with the usage. Template:
def initialization():
  # Getting arguments and options with argparse
  # Initialization of the argparse class
  parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
  # Mandatory parameters
  ## Input file. Try to open it to check if it exists
  parser.add_argument('input',  type=argparse.FileType('r'), default=None, help="Configuration file. Must contain a pair of freq_path dict_path in each line.")
  ## Output file. Try to open it to check if it exists or can be created
  parser.add_argument('output', type=argparse.FileType('wb+'), default=None, help="Merged probabilistic dictionary.")
  parser.add_argument('--stopwords', type=argparse.FileType('w+'), default="stopwords", help="File with stopwords", required=False)  
  parser.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")
  parser.add_argument('-g', '--gzipped', action='store_true', help="Compresses the output file")
  ## Parameters required
  #groupM = parser.add_argument_group('mandatory arguments')
  #groupM.add_argument('-s', '--source_lang', required=True, help="Source language of the input")
  #groupM.add_argument('-t', '--target_lang', required=True, help="Target language of the input")
  
  # Options group
  groupO = parser.add_argument_group('options')
  groupO.add_argument('-s', '--stopwords_amount',  type=int, default=0, help="Amount of words to mark as stopwords")
  groupO.add_argument('-n', '--prune_ratio', type=float, default=10, help="Ratio to prune the dictionary. Translations whose probability is {} times (default) than the maximum one.".format(10))
  groupO.add_argument('-f', '--cutoff_freq',  type=int, default=1, help="Cutoff frequency for merged dictionary (all those equal or below are removed)")
  groupO.add_argument('-k', '--keep_tmp', action='store_true', default=False, help="This flag specifies whether removing temporal folder or not")
  groupO.add_argument('-m', '--tmp_dir', type=check_if_folder, default=gettempdir(), help="Temporary directory where creating the temporary files of this program")

  # Logging group
  groupL = parser.add_argument_group('logging')
  groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
  groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
  groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
  
  # Validating & parsing
  args = parser.parse_args()
  logging_setup(args)

  # Extra-checks for args here
  if (args.prune_ratio != 0):
    args.prune_ratio = math.log(args.prune_ratio)

  return args

def load_freqs(fname):
    result = {}
    with open(fname, "r") as freqs:
      for i in freqs:
        parts = i.strip().split()
        if len(parts)==2:
            n = int(parts[0])
            w = parts[1]
            result[w] = n
        else:
            logging.warning("WRONG FREQ LINE: " + i)
    return result
    

def find_stopwords(final_freqs, swords, swords_file):
    stopwords = []
    sorted_final_freqs = sorted(final_freqs.items(), key=operator.itemgetter(1), reverse=True)
    #logger.debug("Sorted final freqs: " + str(sorted_final_freqs))
    
   # with open(swords_file, 'w+') as swf:        
    removed_words = 0
    while (removed_words<swords):
#   	logger.debug(sorted_final_freqs[removed_words])
        stopwords.append(sorted_final_freqs[removed_words][0]) #adding only key to the stopwords array
        swords_file.write(sorted_final_freqs[removed_words][0])
        swords_file.write(" ")
        swords_file.write(str(sorted_final_freqs[removed_words][1]))
        swords_file.write("\n")
        removed_words += 1
    return stopwords


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
            output.write(t + " " + source + " " + p + "\n")
        else:
            logging.debug("Translation {} -> {} discarded. Prob: {} - Max prob: {}".format(source, t, p, math.exp(threshold)))
            
                                      
# Prune bilingual probabilistic dictionary
# The method keeps only translations whose probability is 10 times (by default) lower than the maximum one
def prune_dictionary(dictionary, args, output):
    logging.debug("Pruning dictionaries")
    current_source = None
    current_translations = []
    for line in dictionary:  
        parts = line.strip().split(" ")
        if current_source != None and parts[1] != current_source:
            prune_translations(args.prune_ratio, current_source, current_translations, output)
            current_translations = []
        current_source = parts[1]
        if (float(parts[2]) != 0):
            current_translations.append((parts[0], parts[2], math.log(float(parts[2]))))
    if len(current_translations) > 0:
        prune_translations(args.prune_ratio, current_source, current_translations, output)
                                                                                                                                                                
                                                                                                                                                                
def main(args):                                                                 
    config_file = args.input
    f_dict = args.output
    swords = args.stopwords_amount
    swords_file = args.stopwords

    temp_file = tempfile.NamedTemporaryFile(mode="wt+", delete=(not args.keep_tmp), dir=args.tmp_dir)
    notpruned_dict = tempfile.NamedTemporaryFile(mode="wt+", delete=(not args.keep_tmp), dir=args.tmp_dir)
    afterpruning_dict = tempfile.NamedTemporaryFile(mode="wt+", delete=(not args.keep_tmp), dir=args.tmp_dir)

    final_freqs = {}
    stopwords = []

    for line in config_file:
        line_parts = line.split()
        f = line_parts[0].strip()
        d = line_parts[1].strip()   
   
        freqs = load_freqs(f)
        with open(d, "r") as f_dic:
            for e in f_dic:
                parts = e.split()
                w1 = parts[0] #target
                w2 = parts[1] #source
                
                # noise removal
                
                if not re.match('^[-\w\']+$', w1) or not re.match('^[-\w\']+$', w2):
                    continue
                    
                if "NULL" in [w1, w2]:
                    continue
                     
                try:
                    rf = float(parts[2])
                except:
                    logging.warning("No frequency for pair {0} - {1}".format(w2, w1))
                    rf = 0
                if w2 in freqs:
                    af = max(1, round(rf*freqs[w2]))                 
                    #af = math.ceil(rf*freqs[w2])
                    
                    if w2.lower() not in final_freqs:
                        final_freqs[w2.lower()] = 0

                    final_freqs[w2.lower()] += af
                    temp_file.write(w2.lower())
                    temp_file.write(" ")
                    temp_file.write(w1.lower())
                    temp_file.write(" ")
                    temp_file.write("{0}\n".format(af))
                else:
                    logging.warning("NO FREQUENCY FOR: {0}".format(w2))
   
    if swords > 0:
        stopwords = find_stopwords(final_freqs, swords, swords_file)                 
        #logger.debug("STOPWORDS: " + str(stopwords))
  
    temp_file.flush()

    
    sort_command = "LC_ALL=C sort {0} -o {0}".format(temp_file.name) # OJO, QUE ESTO ES LEGAL
    p = subprocess.Popen(sort_command, shell=True, stdout=subprocess.PIPE)
    #p = subprocess.Popen(sort_command, shell=True)
    p.wait()
    
    temp_file.seek(0)

    mystack = []

    for e in temp_file:
        #logger.debug("TEMP FILE: " + e)
        parts = e.split()
        w1 = parts[0] #source
        w2 = parts[1] #target
        af = int(parts[2])

        #if w2 in stopwords:
            #logger.debug("Ignoring stopword: " + w2)
            #continue
        if w1 not in final_freqs or final_freqs[w1] <= args.cutoff_freq:  #Final freq of the source!
            logging.debug("Ignoring {0}: abs.freq.={1}".format(w1, final_freqs.get(w1)))
            continue        
        
        if len(mystack) > 0:
            pw1, pw2, paf = mystack.pop()

            if pw1 == w1 and pw2 == w2:
                mystack.append([pw1, pw2, af+paf])
                #logging.debug("Adding {0}, {1}, {2}".format(w1, w2, af+paf))
            else:
                #logger.debug("Writing {0}, {1}, {2}".format(pw1, pw2, float(paf)/final_freqs[pw2.lower()]))               
                notpruned_dict.write(pw2)
                notpruned_dict.write(" ")
                notpruned_dict.write(pw1)
                notpruned_dict.write(" ")
                if pw1 in final_freqs:
                    notpruned_dict.write("{0:1.7f}\n".format(float(paf)/final_freqs[pw1]))    
                else:
                    notpruned_dict.write("0.0000000\n")    
        if len(mystack) == 0:
            mystack.append([w1, w2, af])
            #logging.debug("Appending {0}, {1}, {2}".format(w1, w2, af, mystack))
    else:
        if w1 not in final_freqs or final_freqs[w1] <= args.cutoff_freq:  #Final freq of the source!
            logging.debug("Ignoring {0}: abs.freq.={1}".format(w1, final_freqs.get(w1)))
        else:   
            if len(mystack) > 0:
                pw1, pw2, paf = mystack.pop()
                notpruned_dict.write(pw2)
                notpruned_dict.write(" ")
                notpruned_dict.write(pw1)
                notpruned_dict.write(" ")
                if pw1 in final_freqs:
                    notpruned_dict.write("{0:1.7f}\n".format(float(paf)/final_freqs[pw1]))
                else:
                    notpruned_dict.write("0.0000000\n")

    
    notpruned_dict.seek(0)    

    if args.prune_ratio > 0:
        logging.debug("Pruning")        
        prune_dictionary(notpruned_dict, args, afterpruning_dict)
    else:      
        #Return dict as it is
        logging.debug("Not pruning")
        for i in notpruned_dict:
            afterpruning_dict.write(i)
#    f_dict.seek(0)
    
#    f_dict.close()
    afterpruning_dict.flush()
    temp_file_name = afterpruning_dict.name   
    
    if args.gzipped:
        logging.debug("Return gzipped")
        #f_dict.close()        
        #afterpruning_dict.close()
        #afterpruning_dict.seek(0)
        with open(temp_file_name, 'rb') as ngzd:
            with gzip.open(f_dict, 'wb') as gzd:
                shutil.copyfileobj(ngzd, gzd)
    else:
        logging.debug("Not gzipped")
        #f_dict.close()
        with open(temp_file_name, 'r') as ngzd:
            with open(f_dict.name, 'wb') as gzd:
                shutil.copyfile(temp_file_name, f_dict.name)
#                for i in ngzd:
#                    gzd.write(i)
#       f_dict.close()

 


if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization() # Parsing parameters
        logging_setup(args)
        main(args)  # Running main program
        logging.info("Program finished")
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)

