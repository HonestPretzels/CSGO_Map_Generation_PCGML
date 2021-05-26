from bs4 import BeautifulSoup
from zipfile import ZipFile
import requests
import sys, re
from tqdm import tqdm

def get_6v6_Matches(url, pageNum, outPath):
    ids = []
    maps = set([])

    page = requests.get(url  + '/?p=' + pageNum)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find(class_='table loglist')
    listItems = table.find_all('tr')[1:] # Ignore title column

    for item in listItems[1:]:
        itemFields = item.find_all('td')
        if '6v6' == itemFields[2].text:
            id = item.attrs['id']
            ids.append(id)

            # THIS IS TEMPORARY. NEED TO GET A FEEL FOR NUMBER OF MAPS
            map = itemFields[1].text
            map = re.sub(' ', '', map)
            maps.add(map)

            zippedLog = requests.get(url + '/logs/' + id + '.log.zip').content
            logFileName = outPath + '/' + id + '_' + map + '.log'
            with open('tempFile.zip', 'wb') as tempZip:
                tempZip.write(zippedLog)
            zf = ZipFile('tempFile.zip')
            zinfos = zf.infolist()
            for zi in zinfos:
                zi.filename = logFileName
                zf.extract(zi)

    return ids, maps


def main():
    BaseURL = 'https://logs.tf'
    ids = []
    maps = set([])

    logPath = sys.argv[1]

    pageNum = 1
    pbar = tqdm(total=10000)
    while len(ids) < 10000 and pageNum < 116790:
        foundIds, foundMaps = get_6v6_Matches(BaseURL, str(pageNum), logPath)
        maps.update(foundMaps)
        ids.extend(foundIds)
        pageNum += 1
        pbar.update(len(ids) - pbar.n)
    pbar.close()

    with open('maps.txt', 'w') as f:
        for map in maps:
            f.write(map + '\n')

if __name__ == "__main__":
    main()