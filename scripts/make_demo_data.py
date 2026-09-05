from pathlib import Path
import random
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import extract_one

OUT = ROOT / "data" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(2026)

weak = ["demo123", "sample123", "qwerty77", "abc12345", "welcome9", "guest001", "testtest", "student1"]
mid = ["BlueRiver27", "AdminDemo_2026", "CqStudent88", "Login-Frame77", "SafeDemo520"]
strong = ["Maple!River_47Orbit", "Quartz#91-Lantern", "Cedar^Moon_582", "Velvet+Comet_731"]

def strength(p, domain):
    f = extract_one(p)
    val = 0.38*f['length'] + 2.0*f['entropy'] + 2.6*f['unique_ratio'] + 1.2*f['class_transitions']
    val -= 4.2*f['keyboard_walk_ratio'] + 3.4*f['sequential_ratio'] + 2.4*f['repeated_bigram_ratio']
    val -= 2.8*f['numeric_suffix_signal']
    if domain == 'demoB':
        val += 0.8*f['special_ratio'] - 0.7*f['digit_ratio']
    if domain == 'demoC':
        val += 1.2*f['upper_ratio'] - 0.8*f['year_signal']
    return val + random.gauss(0, 0.65)

for domain in ['demoA','demoB','demoC']:
    rows=[]
    for i in range(260):
        r=random.random()
        if r < .50:
            p=random.choice(weak)
            if random.random()<.3: p += str(random.randint(0,9))
        elif r < .82:
            p=random.choice(mid)
            if random.random()<.35: p += random.choice(['!','@','#'])
        else:
            p=random.choice(strong)
            if random.random()<.25: p += chr(65+random.randint(0,25))
        y = strength(p, domain)
        rows.append({'password':p,'strength':y,'weight':float(random.choice([1,1,1,2,3]))})
    pd.DataFrame(rows).to_csv(OUT/f'result_{domain}_demo.csv', index=False)
print(f'Demo data written to {OUT}')
