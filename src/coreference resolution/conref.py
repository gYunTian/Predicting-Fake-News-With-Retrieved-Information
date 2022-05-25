# this file is used to apply coference resolution on each line in a .txt file
# to adapt this code for use, change the input and output path for the process function

from allennlp.predictors.predictor import Predictor
import spacy
import gc
import torch
import re
import nltk
import timeit

# spacy.require_gpu()
nlp_trf_all = spacy.load('en_core_web_trf')
predictor = Predictor.from_path("../resources/coref-spanbert-large-2021.03.10.tar.gz", cuda_device=0) 

def clear_mem():
  word = None
  gc.collect()
  torch.cuda.empty_cache() 

def write(out, word):
  with open(out, 'a', encoding='utf-8') as out:
    out.write(word)

def coref_replace(doc):
  res = predictor.predict(document=doc)

  word = res['document']
  for i in res['clusters']:
    prev = ""
    entity = ""
    for k in i[0]:
      if (prev.strip() != res['document'][k].strip()): entity += res['document'][k].strip() + " "
      prev = entity
    
    doc = None
    gc.collect()
    torch.cuda.empty_cache() 

    for k in i[1:]:
      prev = 0
      for d in k:
        if (prev != d):
          doc = nlp_trf_all(word[d])
          tag = doc[0].pos_
          if (tag == 'NOUN' or (word[d].strip() == entity.strip())): continue
          word[d] = entity.strip()
        prev = d
  word = ' '.join(word).strip()
  return re.sub(r'\b(\w+) \1\b', r'\1', word, flags=re.IGNORECASE) # remove duplicates in adjacent

def process(input, output, error):
  file = open(input, 'r')
  file = file.read()
  file = file.split('</end>')
  count = 0
  
  for doc in file:
    count += 1
    # if (count < 100): continue
    if (count%100==0): print(count) 
    if (len(set(doc)) > 4):
      length = len(doc)
      if (length > 1000):
        doc = nltk.sent_tokenize(doc)
        try:
          start = 0
          for end in range(3, len(doc), 2):
            word = coref_replace("".join(doc[start:end]))
            write(output, word)
            start += 2 # trailing window
            clear_mem()
        except:
          write(error, ''.join(doc))
          write(error, '\n</end>\n')
          continue
      else:
        try:
          word = coref_replace(doc)
          write(output, word)
          clear_mem()
        except:
          write(error, ''.join(doc))
          write(error, '\n</end>\n')
          continue
      write(output, '\n</end>\n')

if __name__ == "__main__":
  print("------ Starting Conreference Resolution ------")
  for i in range(17,31):
    starttime = timeit.default_timer()
    print("writing to "+'../output_fixed/extracted'+str(i)+'.txt')
    process('../output_fixed/extracted'+str(i)+'.txt', '../output_fixed/conrefed_'+str(i)+'.txt', '../output_fixed/conrefed_'+str(i)+'_error.txt')
    print(str(i)+" took :", timeit.default_timer() - starttime, " seconds")
  print("------ Conreference Resolution done ------")
