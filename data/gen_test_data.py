fout = open('newTestData.tsv','w')
fout.write('id\tsentiment\treview\n')
for idx, line in enumerate(open('test-pos.txt')):
    fout.write('"pos_%s"\t1\t"%s"\n' % (idx, line.rstrip().replace('\t', ' ')))
for idx, line in enumerate(open('test-neg.txt')):
    fout.write('"neg_%s"\t0\t"%s"\n' % (idx, line.rstrip().replace('\t', ' ')))
