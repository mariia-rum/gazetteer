import yaml
import pandas as pd
from .preprocess_messy import preprocess


def get_messy(file_path):
    df_messy = pd.read_csv(file_path).drop_duplicates()
    df_messy = pd.DataFrame(preprocess(df_messy['title']).dropna())
    messy = {i: {'title': row.title} for i, row in enumerate(df_messy.itertuples())}
    return messy


def get_canonical(file_path, category):
    with open(file_path, 'r') as f:
        canonical = yaml.full_load(f)
    canonical = {i: {'title': title.strip().lower()} for i, title in enumerate(canonical[category])}
    return canonical


def get_test(file_path, category):
    with open(file_path, 'r') as f:
        test = yaml.full_load(f)
    test = test[category]
    test = {entity.lower().strip(): {i: {'title': samp.lower().strip()} for i, samp in enumerate(samples)} for entity, samples in test.items()}
    return test
