def saveToFile(sentence, tipe):
    documents_clean = []
    for d in sentence:
        document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
        document_test = re.sub(r'@\w+', '', document_test)
        document_test = document_test.lower()
        document_test = re.sub(r'[\'\"\:\\\/,\(\)\{\}]', ' ', document_test)
        document_test = re.sub(r'[0-9]', '', document_test)
        document_test = re.sub(r'\s{2,}', ' ', document_test)
        documents_clean.append(document_test)

    f = open('C:/Users/angger/PycharmProjects/UTS_Pemerolehan Informasi B/No_1/' + tipe + '.txt', 'a')

    for p in documents_clean:
        p = p + "\n"
        f.write(p)

    f.close()


import requests  # for making HTTP requests in Python
from bs4 import BeautifulSoup  # pulling data from HTML or XML files
import re

r = requests.get('https://kompas.com/sains')
soup = BeautifulSoup(r.text, "html.parser")
print(soup.prettify())
link = []

for i in soup.find('div', {'class': 'mostList -mostlist'}).find_all('a'):
    i['href'] = i['href'] + '?page=all'
    link.append(i['href'])

l = 1
for j in link:
    r = requests.get(j)
    soup = BeautifulSoup(r.content, 'html.parser')
    sen = []

    for k in soup.find('div', {'class': 'read__content'}).find_all('p'):
        sen.append(k.text)

    saveToFile(sen, 'sains' + str(l))
    l = l + 1
