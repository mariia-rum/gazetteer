import os
import dedupe
import pandas as pd

import src.data as data
from src.utils import parse_args
    

if __name__ == '__main__':
    # ## SETTING UP
    DIR_INPUT = 'input'
    DIR_OUTPUT = 'output'
    FILE_CANON = os.path.join(DIR_INPUT, 'cats.yaml')
    FILE_TEST = os.path.join(DIR_INPUT, 'test.yaml')

    
    # parse args
    args = parse_args(FILE_CANON)
    category = args['Category']
    
 
    # setup dependent paths
    DIR_OUTPUT_CATEGORY = os.path.join(DIR_OUTPUT, category)
    FILE_SETTINGS = os.path.join(DIR_OUTPUT_CATEGORY, 'example_learned_settings')
    FILE_TRAINING = os.path.join(DIR_OUTPUT_CATEGORY, 'example_training.json')
    
    
    # ## LOAD DATA
    # test
    test = data.get_test(FILE_TEST, category)
    
    # canonical
    canonical = data.get_canonical(FILE_CANON, category)
    
    
    # ## LOAD GAZETTEER
    with open(FILE_SETTINGS, 'rb') as f:
        gazetteer = dedupe.StaticGazetteer(f)
    gazetteer.index(canonical)
    
    
    # ## EVALUATE
    for entity, samples in test.items():
        print(f'entity: {entity}')

        results = gazetteer.search(samples, n_matches=3, generator=False)

        for smaple_id, ranking in results:
            print(f"\ttest sample: {samples[smaple_id]['title']}")
                
            for entity_id, score in ranking:
                print(f"\t\tcandidate: {canonical[entity_id]['title']}, confidence: {100*score:.2f}")
