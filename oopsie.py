import re
import os
dirname = "test"

regex = re.compile("(.*)_(\d\d\.xml)")
for fname in os.listdir(dirname):
    print(fname)
    match = regex.match(fname)
    onset = match.group(1)
    coda = match.group(2)
    print(onset+coda)
    os.rename(os.path.join(dirname,fname), os.path.join(onset+coda))