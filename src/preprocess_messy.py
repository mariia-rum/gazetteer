import numpy as np


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
        .str.replace(r'[^A-Za-z0-9 ]', '', regex=True)
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
    r"exec?": "executive",
    r"cmo": "chief marketing officer",
    r"cto": "chief technology officer",
    r"ceo": "chief executive officer",
    r"cfo": "chief financial officer",
    r"avp": "assistant vice president",
    r"evp": "executive vice president",
    r"svp": "senior vice president",
    r"coo": "chief operating officer",
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
