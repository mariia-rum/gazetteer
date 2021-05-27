import os
import re
import sys
import yaml
import dedupe
import logging
import numpy as np
import pandas as pd

    
def preprocess(titles, max_title_len=60):
    """
    df:  Series of string (object) dtype
    max_title_len: titles with greater length
    
    return:  Series of the same shape 
    """
    titles = titles.copy()    

    # ## basic cleaning
    # filter out too long titles
    titles[titles.str.len() > max_title_len] = np.nan
    
    titles = (
        titles
        .str.lower()
        .str.replace(r'[\/,.&;:()\-]', ' ', regex=True)
        .str.replace(r'[^A-Za-z ]', '', regex=True)
        .str.replace(r' +', ' ', regex=True)
        .str.strip()
    )
    
    # ## advanced cleaning
    # expand shorcuts to full text
    for w_short, w_long in words_expand.items():
        titles = titles.str.replace(w_short, w_long, regex=True)   

    titles[titles == ''] = np.nan
    
    return titles

    
words_expand = {
    r"vp": "vice president",
    r"cmo": "chief marketing officer",
    r"cto": "chief technology officer",
    r"ceo": "chief executive officer",
    r"cfo": "chief financial officer",
    r"avp": "assistant vice president",
    r"evp": "executive vice president",
    r"svp": "senior vice president",
    r"coo": "chief operating officer",
    r"cto": "chief technical officer",
    r"cio": "chief information officer",
    r"cpo": "chief product officer",
    r"cro": "сhief revenue officer",
    r"cxo": "chief experience officer", 
    r"cdo": "chief data officer",
    
    r"pm": "project manager",
    r"gm": "general manager",
    r"hr": "human resources",
    r"it": "information technology",
    r"ai": "artificial intelligence",
    r"sdr": "sales development representative"
}
words_expand = {fr'\b{key}\b': val for key, val in words_expand.items()}


DIR_INPUT = 'input'
DIR_OUTPUT = 'output'
if not os.path.exists(DIR_OUTPUT):
    os.mkdir(DIR_OUTPUT)

FILE_MESSY = os.path.join(DIR_INPUT, 'bq-results-20210527-151051-61oo7h76w577.csv')
FILE_CANON = os.path.join(DIR_INPUT, 'cats.yaml')

FILE_SETTINGS = os.path.join(DIR_OUTPUT, 'example_learned_settings')
FILE_TRAINING = os.path.join(DIR_OUTPUT, 'example_training.json')


if __name__ == '__main__':
    # ## LOGGING
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('logs.log')
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)


    # ## READ & PREPROCESS & TRANSFORM DATA
    # messy data
    logger.info('reading input messy data...')
    df_messy = pd.read_csv(FILE_MESSY).drop_duplicates()

    logger.info('preprocessing input messy data...')
    df_messy = pd.DataFrame(preprocess(df_messy['title']).dropna())
    
    logger.info('transforming input messy data...')
    messy = {i: {'title': row.title} for i, row in enumerate(df_messy.itertuples())}

    
    # canonical data
    logger.info('reading input canonical data...')
    with open(FILE_CANON, 'r') as f:
        canonical = yaml.full_load(f)
        
    logger.info('transforming input canonical data...')
    canonical = {i: {'title': title.strip().lower()} for i, title in enumerate(canonical['Marketing'])}
    
    
    # ## SETUP GAZETTER
    # Define the fields the gazetteer will pay attention to
    fields = [
        {'field': 'title', 'type': 'String'},
#             {'field': 'title', 'type': 'Text', 'corpus': corpus()},
#             {'field': 'description', 'type': 'Text', 'has missing': True, 'corpus': corpus()},
#             {'field': 'price', 'type': 'Price', 'has missing': True}
    ]

    # Create a new gazetteer object and pass our data model to it.
    gazetteer = dedupe.Gazetteer(fields)

    # If we have training data saved from a previous run of gazetteer, look for it an load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    if os.path.exists(FILE_TRAINING):
        logger.info(f'reading labeled examples from training_file: {FILE_TRAINING}')
        with open(FILE_TRAINING, 'rb') as f:
            gazetteer.prepare_training(messy, canonical, training_file=f)
    else:
        gazetteer.prepare_training(messy, canonical)

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
    
    logger.info('FINISHED\n\n\n')
