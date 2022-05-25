# USAGE: python extract_text.py --input ../data/pdf_json --out ../output_fixed --chunk 30
# python extract_text.py --input ../data/pmc_json --out ../output_fixed --chunk 30
# --input is where the dataset to be extracted is stored at (only works for COV open research dataset downloaded from Kaggle)
# --out is the path to output the extracted data
# this file will run parallely and store the extracted data into different chunks
# the outout is in .txt form where each row is a paragraph


import os
import json
import argparse

from py import process
import preprocess as pp
import timeit
from multiprocessing.pool import ThreadPool as Pool

parser = argparse.ArgumentParser(description='Text extraction cli')
parser.add_argument('--input', type=str, help='specify the json path')
parser.add_argument('--output', type=str, help='specify the output path')
parser.add_argument('--chunk', type=str, help='specify the number of chunks to split corpus into')
args = parser.parse_args()

def worker_extract(task):
  filename, output, iteration = task
  
  with open(os.path.join(input_path, filename), 'r') as f:
    data = json.load(f)
    with open(output, 'a', encoding='utf-8') as out:
      try:
        for line in data['abstract']:
          processed_text = pp.f_base(line['text'])
          out.write(processed_text + "\n")
        # else:
        #   for line in data['body_text']:
        #     processed_text = pp.f_base(line['text'])
        #     out.write(processed_text + "\n")
        out.write('</end>\n')
      except:
        # out.write('</end>\n')
        next
      
  if (iteration%10000==0): print("------- Extracted: " + str(iteration) + " -------")

if __name__=="__main__":
  if args.input:
    print("------- Starting Preprocess parallely --------")
    extracted = list()
    input_path = args.input
    output_path = args.output 
    chunks = int(args.chunk)

    if not os.path.exists(output_path): 
      os.makedirs(output_path)
    assert os.path.exists(input_path)
    
    outputs = [os.path.join(output_path, 'extracted'+str(i)+'.txt') for i in range(31,31+chunks+1)]

    tasks = list()
    i = 0
    for filename in os.listdir(input_path):
      tasks.append((filename, outputs[i%chunks], i))
      i += 1
    outputs = None

    starttime = timeit.default_timer()
    pool = Pool(processes=12)
    pool.map(worker_extract, tasks)
    pool.close()
    pool.join()
    print("------- Preprocess Complete --------")
    print("Preprocessing took :", timeit.default_timer() - starttime, " seconds")
    exit()
