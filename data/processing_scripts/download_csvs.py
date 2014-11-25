import urllib2

with open('./stats_page.txt', 'rb') as f:
    content = f.readlines()
    for line in content:
        if '.csv</a>' in line:
            line = line.split(".csv'>")[1].split('<')[0]
            response = urllib2.urlopen('http://www.basketballgeek.com/downloads/2009-2010/'+line)
            output_csv = response.read()
            o = open('./'+line,'w')
            o.write(output_csv)
            o.close()
