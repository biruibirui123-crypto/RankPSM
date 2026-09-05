import math, re
from collections import Counter
import pandas as pd

QWERTY_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
ADJ=set()
for row in QWERTY_ROWS:
    for a,b in zip(row,row[1:]):
        ADJ.add((a,b)); ADJ.add((b,a))
LEET=set("013457@$!")

def shannon_entropy(s):
    s = "" if s is None else str(s)
    if not s: return 0.0
    c=Counter(s); n=len(s)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def max_run(s):
    s=str(s)
    if not s: return 0
    best=cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1; best=max(best,cur)
        else: cur=1
    return best

def cat(ch):
    if ch.islower(): return 0
    if ch.isupper(): return 1
    if ch.isdigit(): return 2
    return 3

def transitions(s):
    s=str(s)
    return sum(cat(a)!=cat(b) for a,b in zip(s,s[1:])) if len(s)>1 else 0

def keyboard_ratio(s):
    s=str(s).lower()
    if len(s)<2: return 0.0
    return sum((a,b) in ADJ for a,b in zip(s,s[1:]))/(len(s)-1)

def sequential_ratio(s):
    s=str(s).lower()
    if len(s)<2: return 0.0
    hits=0
    for a,b in zip(s,s[1:]):
        if a.isdigit() and b.isdigit() and abs(ord(a)-ord(b))==1: hits+=1
        elif a.isalpha() and b.isalpha() and abs(ord(a)-ord(b))==1: hits+=1
    return hits/(len(s)-1)

def repeated_bigram_ratio(s):
    s=str(s)
    if len(s)<4: return 0.0
    grams=[s[i:i+2] for i in range(len(s)-1)]
    cnt=Counter(grams)
    rep=sum(v-1 for v in cnt.values() if v>1)
    return rep/max(1,len(grams))

def extract_one(s):
    s="" if s is None else str(s); n=len(s)
    lower=sum(c.islower() for c in s); upper=sum(c.isupper() for c in s); digit=sum(c.isdigit() for c in s)
    alpha=lower+upper; special=n-alpha-digit; uniq=len(set(s))
    ent=shannon_entropy(s)
    return {
        'length':n,
        'lower_count':lower, 'upper_count':upper, 'digit_count':digit, 'special_count':special,
        'lower_ratio':lower/max(1,n), 'upper_ratio':upper/max(1,n), 'digit_ratio':digit/max(1,n), 'special_ratio':special/max(1,n),
        'unique_ratio':uniq/max(1,n), 'entropy':ent, 'entropy_x_len':ent*n,
        'max_run':max_run(s), 'class_transitions':transitions(s),
        'keyboard_walk_ratio':keyboard_ratio(s), 'sequential_ratio':sequential_ratio(s), 'repeated_bigram_ratio':repeated_bigram_ratio(s),
        'year_signal':int(bool(re.search(r'(19\d{2}|20\d{2})',s))),
        'numeric_suffix_signal':int(bool(re.search(r'(123|1234|12345|000|111|666|888|520|1314)$',s.lower()))),
        'leet_ratio':sum(c in LEET for c in s)/max(1,n),
        'starts_upper':int(bool(s) and s[0].isupper()), 'ends_digit':int(bool(s) and s[-1].isdigit()),
    }

def frame(passwords):
    return pd.DataFrame([extract_one(x) for x in passwords])
