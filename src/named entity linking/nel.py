# this file is used to perform named entity linking
# it works on .txt file row by row
# likewise, change the with open paths to where the .txt file is stored

import urllib.request
import json
import timeit
import spacy
import re
import os
import multiprocessing
from multiprocessing.pool import ThreadPool as Pool

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')
key = "wlyydzcwcckwtxorwnnxscvtmrtqxc"

# http://wikifier.org/info.html


def get_wikifier(text, key, url="http://www.wikifier.org/annotate-article", lang="en", threshold=0.9, timeout=10):
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", key),
        ("pageRankSqThreshold", "%g" %
         threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
    ])

    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    response = None
    with urllib.request.urlopen(req, timeout=timeout) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    return response


def extract_entity(response, text):
    global unique
    results = list()
    unique = set()

    for annotation in response['annotations']:
        if ('wikiDataClasses' in annotation):
            if (len(annotation['wikiDataClasses']) > 0):
                if (annotation['support'][0]['prbConfidence'] >= 0.5):

                    start = annotation['support'][0]['chFrom']
                    end = annotation['support'][0]['chTo']
                    entity = text[start:end+1].strip()
                    if (entity not in unique):
                        unique.add(entity)
                        doc = nlp(entity)  # identify pos
                        tag = doc[0].pos_

                        entity_type = annotation['wikiDataClasses'][0]['enLabel']
                        for match in re.finditer(entity, text):
                            results.append({"entity": entity, "type": entity_type, "start": match.start(
                            ), "end": match.end(), "tag": tag})

    with open(os.path.join('../output_post/', 'extracted.txt'), 'a', encoding='utf-8') as out:
        output = str(text) + "<sep>" + str(results) + "</end>\n"
        out.write(output)

def worker_wiki(text):
    try:
        response = get_wikifier(text=text, key=key)
        extract_entity(response, text)
    except Exception as e:
        print(e)
        with open(os.path.join('../output_post/', 'nel_error.txt'), 'a', encoding='utf-8') as out:
            out.write(text + "</end>\n")

if __name__ == "__main__":
    print("------ Starting NEL ------")
    for i in range(3, 31):
        try:
            tasks = list()
            with open(os.path.join('../output_post/', 'new_'+str(i)+'.txt'), 'r') as f:
                print("NEL doc", i)
                text = f.read()
                data = text.split('</end>')
                for line in data:
                    tasks.append(line)

            starttime = timeit.default_timer()
            pool = Pool(processes=16)
            pool.map(worker_wiki, tasks)
            pool.close()
            pool.join()
            print("NEL took :", timeit.default_timer() - starttime, " seconds")
        except:
          continue
    print("------ NEL Complete ------")
