import argparse
import fileinput
import random
import torch

#cat monolingual.de | python $BPEROOT/apply_bpe.py -c code | python interactive.py $DATA --path model.pt --buffer-size 1024 --beam 5 \
#--batch-size 16 |grep -P '^H' |cut -f3- | sed 's/@@\s*//g' | python addnoise.py > translation.en

def main():
    parser = argparse.ArgumentParser(description='Command-line script to add noise to data')
    parser.add_argument('-wd', help='Word dropout', default=0.1, type=float)
    parser.add_argument('-wb', help='Word blank', default=0.1, type=float)
    parser.add_argument('-sk', help='Shufle k words', default=3, type=int)
    args = parser.parse_args()
    wd = args.wd
    wb = args.wb
    sk = args.sk

    for s in fileinput.input('-'):
        s = s.strip().split()
        if len(s) > 0:
            s = word_shuffle(s, sk)
            s = word_dropout(s, wd)
            s = word_blank(s, wb)
        print(' '.join(s))



def word_shuffle(s, sk):
    noise = torch.rand(len(s)).mul_(sk)
    perm = torch.arange(len(s)).float().add_(noise).sort()[1]
    return [s[i] for i in perm]

def word_dropout(s, wd):
    keep = torch.rand(len(s))
    res = [si for i,si in enumerate(s) if keep[i] > wd ]
    if len(res) == 0:
        return [s[random.randint(0, len(s)-1)]]
    return res

def word_blank(s, wb):
    keep = torch.rand(len(s))
    return [si if keep[i] > wb else 'Ж' for i,si in enumerate(s)]

if __name__ == '__main__':
    main()
    
    