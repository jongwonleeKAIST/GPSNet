import csv
import sys

# please specify raw csv dataset name you want to process.)
rawfilename = 'raw_dataset_{}.csv'.format(sys.argv[1])
outputfilename = rawfilename[4:]


outputfile = open(outputfilename, 'w')

with open(rawfilename, 'r') as textfile:
    for line in reversed(list(csv.reader(textfile))): # reversely (in fact, it just read the data 'chronological order') read the raw data
        line = ', '.join(line)
        item = line.split("@")
        try:
            item = item[1].split(" ")
            outputfile.write(item[0] + ',' + item[1] + '\n')
        except IndexError:
            pass
