import yaml
import argparse


def parse_args(file_categories):
    ap = argparse.ArgumentParser()
    
    with open(file_categories, 'r') as f:
        categories = yaml.full_load(f).keys()
        
    ap.add_argument("-c", "--Category", type=str, required=True, choices=list(categories))
    
    args = vars(ap.parse_args())
    return args
