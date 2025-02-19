import os
import sys

for aline in sys.stdin:
    aline = aline.strip()
    cols = aline.split()
    if int(cols[0]) % 400 == 0:
        print(aline)
