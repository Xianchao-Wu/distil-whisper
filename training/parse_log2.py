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

akey = " | Loss: "

iter2loss = dict()

def parse_iter_loss(aline):
    #cols = aline.split(akey)
    key1 = 'Step... '
    key2 = 'Learning Rate: '
    #for i in range(len(cols)):
    
    cols = aline.split(' ')
    for i in range(len(cols)):
        if cols[i] == 'Loss:':
            aloss = 'NA'
            if i+1 < len(cols):
                aloss = cols[i+1].replace(',', '')
                aloss = (float(aloss))
            aiter = 'NA'
            if i-4 >= 0:
                aiter = cols[i-4].replace('(', '')
            if aiter != 'NA' and aloss != 'NA':
                if int(aiter)%200==0:
                    #print(aiter, aloss)
                    if aiter in iter2loss:
                        iter2loss[aiter].append(aloss)
                    else:
                        iter2loss[aiter] = [aloss]

for aline in sys.stdin:
    aline = aline.strip()
    #if 'Train steps ... ' in aline:
    #if aline.startswith('Eval results for step'):
    if akey in aline:
        parse_iter_loss(aline)

for aiter in iter2loss:
    losses = iter2loss[aiter]
    #print(aiter, losses)
    avg = sum(losses)/len(losses)
    print(aiter, avg)

