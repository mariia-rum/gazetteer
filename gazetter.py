import os
import json
import dedupe
import logging
import numpy as np
import pandas as pd

import src.data as data
from src.utils import parse_args


def corpus(messy, canonical):
    for dataset in (messy, canonical):
        for record in dataset.values():
            yield record['title']

            
if __name__ == '__main__':
    # ## SETTIG UP
    # setup paths
    DIR_LOGS = 'logs'
    DIR_INPUT = 'input'
    DIR_OUTPUT = 'output'
    FILE_MESSY = os.path.join(DIR_INPUT, 'bq-results-20210527-151051-61oo7h76w577.csv')
    FILE_CANON = os.path.join(DIR_INPUT, 'cats.yaml')
    
    
    # parse args
    args = parse_args(FILE_CANON)
    category = args['Category']
    
 
    # setup dependent paths
    DIR_OUTPUT_CATEGORY = os.path.join(DIR_OUTPUT, category)
    DIR_LOGS_CATEGORY = os.path.join(DIR_LOGS, category)

    FILE_SETTINGS = os.path.join(DIR_OUTPUT_CATEGORY, 'example_learned_settings')
    FILE_TRAINING = os.path.join(DIR_OUTPUT_CATEGORY, 'example_training.json')
    FILE_LOGS = os.path.join(DIR_LOGS_CATEGORY, 'logs.log') 
    
    
    # create folders
    directories = [DIR_OUTPUT, DIR_OUTPUT_CATEGORY, DIR_LOGS, DIR_LOGS_CATEGORY]
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)
  

    # logging
    logger = logging.getLogger(category)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(FILE_LOGS)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)


    # ## READ & PREPROCESS & TRANSFORM DATA
    # messy data
    logger.info('loading messy data...')
    messy = data.get_messy(FILE_MESSY)
    
    logger.info(f'sample size of messy data: {len(messy)}')
    logger.info(f'sample of messy data: {messy[0]}')

    
    # canonical data
    logger.info('loading canonical data...')
    canonical = data.get_canonical(FILE_CANON, category)
    
    logger.info(f'sample size of canonical data: {len(canonical)}')
    logger.info(f'sample of canonical data: {canonical[0]}')


    # ## SETUP GAZETTER
    # Define the fields the gazetteer will pay attention to
    fields = [
        {'field': 'title', 'type': 'Exact'},
        {'field': 'title', 'type': 'String'},
        {'field': 'title', 'type': 'Text', 'corpus': corpus(messy, canonical)},
    ]

    # Create a new gazetteer object and pass our data model to it.
    gazetteer = dedupe.Gazetteer(fields)

    # If we have training data saved from a previous run of gazetteer, look for it an load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    params = {
        'blocked_proportion': 0.5
    }
    
    if os.path.exists(FILE_TRAINING):
        logger.info(f'reading labeled examples from training_file: {FILE_TRAINING}')
        with open(FILE_TRAINING, 'rb') as f:
            gazetteer.prepare_training(messy, canonical, training_file=f, **params)
    else:
        gazetteer.prepare_training(messy, canonical, **params)

    # ## Active learning
    # Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as matches or not.
    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    logger.info('starting active labeling...')
    dedupe.console_label(gazetteer)

    # Using the examples we just labeled, train the gazetteer
    logger.info('training gazetteer...')
    gazetteer.train()

    # When finished, save our training away to disk
    logger.info(f'saving training file: {FILE_TRAINING}')
    with open(FILE_TRAINING, 'w') as f:
        gazetteer.write_training(f)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    logger.info(f'saving settings file: {FILE_SETTINGS}')
    with open(FILE_SETTINGS, 'wb') as f:
        gazetteer.write_settings(f)

    # Clean up data we used for training. Free up memory.
    gazetteer.cleanup_training()

    # Add records to the index of records to match against. 
    # If a record in canonical_data has the same key as a previously indexed record, 
    # the old record will be replaced. 
    gazetteer.index(canonical)
    
    # log statistics
    with open(FILE_TRAINING, 'r') as f:
        votes = json.load(f)
        n_yes = len(votes['match']) 
        n_no = len(votes['distinct'])
        logger.info(f'number of matches: {n_yes}')
        logger.info(f'number of distinct: {n_no}')
        logger.info(f'number of total labeled pairs: {n_yes+n_no}')
    
    logger.info('FINISHED\n\n\n')
