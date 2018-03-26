# Bicleaner

## Installation instructions

### Prerequisites

Bicleaner works with Python 3 only.

#### Python

You will need to install (i.e. with `pip`):

```shell
$ pip install -r requirements.txt
```

#### Other

Also, you would need to install the [**Moses**](http://www.statmt.org/moses/)
decoder somewhere in your system in order to build your own dictionaries. 
Please follow the instructions 
[here](http://www.statmt.org/moses/?n=Development.GetStarted).

You would also need to install [mgiza](https://github.com/moses-smt/mgiza/) 

## Build dictionaries

bicleaner-merge-dictionaries.py is a Python script that allows to merge several probabilistic dictionaries, given the single probabilistic dictionary and absolute frequency for each corpus.
This script is located in bicleaner-filter/src, and can be run with

```shell 
$ python3 bicleaner-merge-dictionaries.py [-h] [--stopwords STOPWORDS] [-v] [-g]
                                       [-s STOPWORDS_AMOUNT] [-n PRUNE_RATIO]
                                       [-f CUTOFF_FREQ] [-k] [-m TMP_DIR] [-q]
                                       [--debug] [--logfile LOGFILE]
                                       input output



```

### Parameters

* positional arguments:
  * input: Configuration file. Must contain a pair of "freq_path dict_path" in each line.
  * output: Merged probabilistic dictionary. Will be created if not exists.
* optional arguments:
  * -h, --help: show this help message and exit
  * --stopwords STOPWORDS: File with stopwords (output)
  * -v, --version: show version of this script and exit
  * -g, --gzipped: Compresses the output file
* options:
  * -s STOPWORDS_AMOUNT, --stopwords_amount STOPWORDS_AMOUNT: Amount of words to mark as stopwords
  * -n PRUNE_RATIO, --prune_ratio PRUNE_RATIO: Ratio to prune the dictionary. Translations whose probability is 10 times (default) than the maximum one.
  * -f CUTOFF_FREQ, --cutoff_freq CUTOFF_FREQ: Cutoff frequency for merged dictionary (all those equal or below are removed)
  * -k, --keep_tmp: This flag specifies whether removing temporal folder or not
  * -m TMP_DIR, --tmp_dir TMP_DIR: Temporary directory where creating the temporary files of this program

* logging:
  * -q, --quiet: Silent logging mode
  * --debug: Debug logging mode
  * --logfile LOGFILE: Store log to a file

### Configuration 

For each direction of each pair, a configuration input file must be provided. It will contain the path to a the absolute frequency of words in a corpus, and in the same line separated by a blank space, the path to the probabilistic dictionary for that corpus.
For example, the content in csen.input can be:

```
OPUS-freqs/cs-en/DGT/DGT.cs OPUS-dicts/cs-en/DGT/lex.f2e
OPUS-freqs/cs-en/ECB/ECB.cs OPUS-dicts/cs-en/ECB/lex.f2e
OPUS-freqs/cs-en/EMEA/EMEA.cs OPUS-dicts/cs-en/EMEA/lex.f2e
```

In the same way, for en-cs (the inverse direction) the file encs.input will be:

```
OPUS-freqs/cs-en/DGT/DGT.en OPUS-dicts/cs-en/DGT/lex.e2f
OPUS-freqs/cs-en/ECB/ECB.en OPUS-dicts/cs-en/ECB/lex.e2f
OPUS-freqs/cs-en/EMEA/EMEA.en OPUS-dicts/cs-en/EMEA/lex.e2f
```

#### Frequency files

Frequency files contain a pair of frequency and word in each line, as in the following example:

```
5022361 the
3104934 ,
3077432 of
1967564 .
1623412 and
1613540 to
1578944 in
1296945 )
1293761 (
 873733 for
```

Frequency files for a given corpus can be built from [OPUS](http://opus.nlpl.eu/) tokenized corpora.
In order to do so, go to the OPUS page, select the desired pair, and download the resources provided in the "mono" column (one file for the source language, and another one for the target language)

Once downloaded, frequency files can be built with the following commands (it builts a frequency file for all corpus in a given pair):

```shell
for file in *.gz; do gunzip -d $file; done
for f in download.php\?f\=*; do mv "$f" $(echo $f |  sed 's/^download.php?f=.*2F//g'); done

calc_freqs() {
tr " " "\n" <$1 | grep -v "^$" | LC_ALL=POSIX sort |uniq -c |LC_ALL=POSIX sort -nr >$1.freq
}
export -f calc_freqs
parallel calc_freqs ::: $(find . -type f -name "*.??")
```

#### Dictionary files

Probabilistic dictionary files contain a triplet of target word, source word, and the probability that the target word is a translation of the source word, as in the following example:

```
Vaikovihovou Vaikovih 1.0000000
Estrogeny Oestrogens 1.0000000
ekologií ecology 0.0400000
heliportů heliports 0.3333333
WEIFANG WEIFANG 1.0000000
Korespondenci Correspondence 0.0111111
Hadamar Hadamar 1.0000000
LEASING LEASING 0.2500000
Fatemi Fatemi 1.0000000
žvýkacího jaw 0.0338983
metkefamid metkefamide 1.0000000
```

Probabilistic dictionary files for a given corpus can be downloaded from [OPUS](http://opus.nlpl.eu/).
In order to do so, go to the OPUS page, select the desired pair, click in the link provided in the "alg" column, then click in "model". The needed files are "lex.e2f" and "lex.f2e" (each one is in each direction of the language pair)

### Example

```shell
$ python3 bicleaner-merge-dictionaries.py ../dict-merger/inputfiles/enlv.input ../dict-merger/outputfiles/enlv.pruned \ 
    -s 100 --stopwords ../dict-merger/stopwords/enlv.swords -n 10 -g -f 1
```

This will generate the probabilistic dictionary for en-lv in the en-lv.pruned gzipped file, using the pairs of probabilistic ditionaries and absolute frequencies for each corpus stated in the enlv.input configuration file. It will remove the translations whose probability is 10 times lower than the maximum one, and also the translations with only one occurence. It will also extract the 100 words with the highest frequency, and will output them in the enlv.swords file. 

## Training classifiers

bicleaner-train.py is a Python script that allows to train a classifier which predicts whether a pair of sentences are mutual translations or not.

This script is located in bicleaner-filter/src, and can be run with

```shell
$ python3 bicleaner-train.py [-h] -m METADATA -c CLASSIFIER -s SOURCE_LANG -t
                          TARGET_LANG -d SOURCE_DICTIONARY -D
                          TARGET_DICTIONARY [--normalize_by_length]
                          [--treat_oovs] [--qmax_limit QMAX_LIMIT]
                          [--disable_features_quest]
                          [-g GOOD_EXAMPLES] [-w WRONG_EXAMPLES]
                          [--good_test_examples GOOD_TEST_EXAMPLES]
                          [--wrong_test_examples WRONG_TEST_EXAMPLES]
                          [--classifier_type {svm,nn,nn1,adaboost}]
                          [--dump_features DUMP_FEATURES] [-b BLOCK_SIZE]
                          [-p PROCESSES]
                          [--wrong_examples_file WRONG_EXAMPLES_FILE] [-q]
                          [--debug] [--logfile LOGFILE]
                          [input]
```

### Parameters

* positional arguments:
  * input: Tab-separated bilingual input file (default: <_io.TextIOWrapper name='<stdin>' mode='r'encoding='UTF-8'>)
* optional arguments:
  * -h, --help: show this help message and exit
* Mandatory:
  * -m METADATA, --metadata METADATA: Training metadata (YAML file) (default: None)
  * -c CLASSIFIER, --classifier CLASSIFIER: Classifier data file (default: None)
  * -s SOURCE_LANG, --source_lang SOURCE_LANG: Source language code (default: None)
  * -t TARGET_LANG, --target_lang TARGET_LANG: Target language code (default: None)
  * -d SOURCE_DICTIONARY, --source_dictionary SOURCE_DICTIONARY: LR gzipped probabilistic dictionary (default: None)
  * -D TARGET_DICTIONARY, --target_dictionary TARGET_DICTIONARY: RL gzipped probabilistic dictionary (default: None)
* Options:
  * --normalize_by_length: Normalize by length in qmax dict feature (default: False)
  * --treat_oovs: Special treatment for OOVs in qmax dict feature (default: False)
  * --qmax_limit: Number of max target words to be taken into account, sorted by length (default: 20)
  * --disable_features_quest: Disable less important features (default: True)
  * -g GOOD_EXAMPLES, --good_examples GOOD_EXAMPLES: Number of good examples (default: 50000)
  * -w WRONG_EXAMPLES, --wrong_examples WRONG_EXAMPLES: Number of wrong examples (default: 50000)
  * --good_test_examples GOOD_TEST_EXAMPLES: Number of good test examples (default: 2000)
  * --wrong_test_examples WRONG_TEST_EXAMPLES: Number of wrong test examples (default: 2000)
  * --classifier_type {svm,nn,nn1,adaboost}: Classifier type (default: svm)
  * --dump_features DUMP_FEATURES: Dump training features to file (default: None)
  * -b BLOCK_SIZE, --block_size BLOCK_SIZE: Sentence pairs per block (default: 10000)
  * -p PROCESSES, --processes PROCESSES: Number of process to use (default: 71)
  * --wrong_examples_file WRONG_EXAMPLES_FILE: File with wrong examples extracted to replace the synthetic examples from method used by default (default: None)
* Logging:
  * -q, --quiet: Silent logging mode (default: False)
  * --debug: Debug logging mode (default: False)
  * --logfile LOGFILE: Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)

### Example

```shell
$ python3 bicleaner-train.py \
          data/cs-en/train.cs-en\
          --treat_oovs \
          --normalize_by_length \
          -s cs \
          -t en \
          -d ../dicts/cs-en.dict.gz \
          -D ../dicts/en-cs.dict.gz \
          -b  1000 \
          -c ../classifiers/cs-en.classifier \
          -g 50000 \
          -w 50000 \
          -m ../classifiers/training.cs-en.yaml \
          --classifier_type svm
```

This will train a SVM classifier for czech-english using the corpus train.cs-en and the dictionaries cs-en.dict.gz and en-cs.dict.gz (that can be generated as shown in the previous section). 
This training will use 50000 good and 50000 bad examples, and a block size of 1000 sentences. The classifier data will be stored in cs-en.classifier, with the metadata in training.cs-en.yaml.

The generated .yaml file provides the following information, that is useful to get a sense on how good or bad was the training:

```
classifier: classifiers/cs-en.classifier
classifier_type: svm
source_lang: cs
target_lang: en
source_dictionary: bicleaner-filter/dicts/cs-en.dict.gz
target_dictionary: bicleaner-filter/dicts/en-cs.dict.gz
normalize_by_length: True
treat_oovs: True
qmax_limit: 20
disable_features_quest: True
good_examples: 50000
wrong_examples: 50000
good_test_examples: 2000
wrong_test_examples: 2000
good_test_histogram: [3, 7, 11, 18, 21, 32, 42, 68, 95, 703]
wrong_test_histogram: [1478, 105, 88, 63, 71, 48, 45, 47, 30, 25]
precision_histogram: [0.3333333333333333, 0.6563528637261357, 0.7036247334754797, 0.7484709480122325, 0.7832110839445803, 0.8281938325991189, 0.8606635071090047, 0.8946280991735537, 0.9355216881594373, 0.9656593406593407]
recall_histogram: [1.0, 0.997, 0.99, 0.979, 0.961, 0.94, 0.908, 0.866, 0.798, 0.703]
accuracy_histogram: [0.3333333333333333, 0.825, 0.8576666666666667, 0.8833333333333333, 0.8983333333333333, 0.915, 0.9203333333333333, 0.9213333333333333, 0.9143333333333333, 0.8926666666666667]
length_ratio: 0.9890133482780752
```


## Cleaning

bicleaner-classifier-tabs.py is a Python script that allows to classify a parallel corpus, indicating whether a pair of sentences are mutual translations (marking it as "keep") or not ("marking it as "discard")
This script is located in bicleaner-filter/src, and can be run with

```shell
$ python3 bicleaner-classifier-tabs.py [-h] -m METADATA [--tmp_dir TMP_DIR]
                                    [-b BLOCK_SIZE] [-p PROCESSES]
                                    [--normalize_by_length] [--treat_oovs]
                                    [--qmax_limit QMAX_LIMIT]
                                    [--disable_features_quest]
                                    [-g GOOD_EXAMPLES] [-w WRONG_EXAMPLES]
                                    [--good_test_examples GOOD_TEST_EXAMPLES]
                                    [--wrong_test_examples WRONG_TEST_EXAMPLES]
                                    [-d DISCARDED_TUS] [--threshold THRESHOLD]
                                    [-q] [--debug] [--logfile LOGFILE] [-v]
                                    input [output]
```

### Parameters

* positional arguments:
  * input: Tab-separated files to be classified
  * output: Output of the classification (default: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='UTF-8'>)
* optional arguments:
  * -h, --help: show this help message and exit
* Mandatory:
  * -m METADATA, --metadata METADATA: Training metadata (YAML file). Take into account that explicit command line arguments will overwrite the values from metadata file (default: None)
* Optional:
  * --tmp_dir TMP_DIR: Temporary directory where creating the temporary files of this program (default: user's temp dir)
  * -b BLOCK_SIZE, --block_size BLOCK_SIZE Sentence pairs per block (default: 10000)
  * -p PROCESSES, --processes PROCESSES: Number of processes to use (default: 71)
  * --normalize_by_length: Normalize by length in qmax dict feature (default: False)
  * --treat_oovs: Special treatment for OOVs in qmax dict feature (default: False)
  * --qmax_limit: Number of max target words to be taken into account, sorted by length (default: 20)
  * --disable_features_quest: Disable less important features (default: True)
  * -g GOOD_EXAMPLES, --good_examples GOOD_EXAMPLES: Number of good examples (default: 50000)
  * -w WRONG_EXAMPLES, --wrong_examples WRONG_EXAMPLES: Number of wrong examples (default: 50000)
  * --good_test_examples GOOD_TEST_EXAMPLES: Number of good test examples (default: 2000)
  * --wrong_test_examples WRONG_TEST_EXAMPLES: Number of wrong test examples (default: 2000)
  * -d DISCARDED_TUS, --discarded_tus DISCARDED_TUS: TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file. (default: None)
  * --threshold THRESHOLD: Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default. (default: 0.5)
* Logging:
  * -q, --quiet: Silent logging mode (default: False)
  * --debug: Debug logging mode (default: False)
  * --logfile LOGFILE: Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)
  * -v, --version: show version of this script and exit


### Example

```shell
$ bicleaner-classifier-tabs.py ../corpus/bicleaner-corpus.en-ro ../classifiers/en-ro.classifier.output -m ../classifiers/training.en-ro.yaml 

```

This will classify the corpus bicleaner-corpus.en-ro using the classifier generated in the training step (see previous step) training.en-ro.yaml. The result of the classification will be stored in en-ro.classifier.output.
This output file will have four columns: two with two sentences to classify (the first one in english, the second one in romanian), the score assigned to that pair, and the classifier recomendation on that sentencen (to keep, or to discard)
