# src/util/helpers.py

import numpy as np

def quarter_to_year_q(qstr):
    q = int(qstr[0])           
    yr = int(qstr[2:])         
    year_full = 2000 + yr      
    return year_full * 10 + q  

def exp_weights(n, decay):
    w = np.array([decay**i for i in range(n)][::-1])
    return w / w.sum()

def green(text): 
    return f"\033[92m{text}\033[0m"

def yellow(text): 
    return f"\033[93m{text}\033[0m"

def red(text): 
    return f"\033[91m{text}\033[0m"

def safe_kw(kw: str) -> str:
    return kw.lower().replace(" ", "_")