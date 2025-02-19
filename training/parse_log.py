import os
import sys

keys = [
    'Eval results for step (',
    'Eval Loss: ',
    'Eval cer: ',
    'Eval wer_ortho: '
]
def parse_line(aline):
    items = list()
    for akey in keys:
        if not akey in aline:
            return None

        cols = aline.split(akey)
        #import ipdb; ipdb.set_trace()
        aitem = cols[1].split()[0]
        items.append(aitem)
    return items

#alogfn="3.run.distill.mtp.parallel.sh.log", batchsize=16, n=3

'''
with open(alogfn) as br:
    for aline in br.readlines(): 
        #for aline in sys.stdin:
        aline = aline.strip()
        #if 'Train steps ... ' in aline:
        if aline.startswith('Eval results for step'):
            items = parse_line(aline)
            if items is not None:
                print(' '.join(items))
'''

for aline in sys.stdin:
    aline = aline.strip()
    #if 'Train steps ... ' in aline:
    if aline.startswith('Eval results for step'):
        items = parse_line(aline)
        if items is not None:
            print(' '.join(items))
