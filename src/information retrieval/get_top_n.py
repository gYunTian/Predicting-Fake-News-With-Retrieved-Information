# this file is used to find top 3 similar context to each query
# 

from cProfile import label
import pandas as pd
import numpy as np
import preprocessor as p
import numpy as np
import re, spacy
from nltk.corpus import stopwords
import nltk, time
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from multiprocessing.pool import ThreadPool as Pool
import pandas as pd
import os

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
# p.OPT.NUMBER

def do_process_eng_hashtag(input_text: str) -> str:
    return re.sub(
        r'#[a-z]\S*',
        lambda m: ' '.join(re.findall('[A-Z][^A-Z]*|[a-z][^A-Z]*', m.group().lstrip('#'))),
        input_text,
    )

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    # temp = do_process_eng_hashtag(tweet)
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    # temp = re.sub("@[A-Za-z0-9_]+","", temp)
    # temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    #temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def get_top_n(inc, orig, label):
    s = str(''.join(inc))
    tn = search.get_top_n(s.split(" "), orig, n=3)
    temp = str(tn)
    outer = s + "<sep>" + (str(label)+"<sep>") + (str(tn[0])+"<sep>") + (str(tn[1])+"<sep>") + (str(tn[2])+"<sep>") + ("</end>\n")
    with open(os.path.join('./test_1000.txt'), 'a', encoding='utf-8') as out:
        out.write(outer)

def get_data(skip=False): # get knowledge base
    if (skip):
        data = pd.read_csv("./merged_data.csv")
        data['text'] = data['text'].astype(str)

        orig = data['text'].to_list()
        data['text'] = data.text.apply(lemmatize_text)
        arr = data['text'].to_list()
        data = None
        return orig, arr

    explanation1 = pd.read_csv("./data/data/train3_fake_exp.csv")
    explanation2 = pd.read_csv("./data/data/train4_exp.csv")

    explanation1 = explanation1[['explanation']]
    explanation2 = explanation2[['explanation']]

    explanation1 = explanation1.rename(columns={'explanation': 'text'})
    explanation2 = explanation2.rename(columns={'explanation': 'text'})

    corpus = pd.read_csv("./data/triples.csv")
    corpus['text'] = corpus['subject'] + " " + corpus['predicate'] + " " + corpus['object']
    corpus = corpus[['text']]

    corpus['text'] = corpus['text'].apply(lambda x: x.strip())
    corpus = corpus.drop_duplicates(subset=['text'], keep=False)

    df = pd.read_csv('./data/sentences/extracted_cleaned_1.txt', sep="<sep>", header=None)
    df = df.rename({0: 'text'}, axis=1)
    df = df[['text']]


    for i in range(1,31):
        t = pd.read_csv('./data/sentences/extracted_cleaned_'+str(i)+'.txt', sep="<sep>", header=None)
        t = t.rename({0: 'text'}, axis=1)
        t = t[['text']]
        df = pd.concat([df, t], axis=0)
        
    df['text'] = df['text'].apply(lambda x: str(x).strip())
    df = df.drop_duplicates(subset=['text'], keep=False)
    df['text'] = df['text'].apply(clean_tweet)
    df['text'] = df['text'].apply(p.tokenize)

    data = pd.concat([corpus, df], axis=0)
    corpus = None
    df = None

    data['text'] = data['text'].apply(lambda x: str(x).strip())
    data = data.drop_duplicates(subset=['text'], keep=False)

    # data = pd.concat([data, explanation1, explanation2], axis=0)

    data['text'] = data['text'].apply(lambda x: str(x).strip())
    data = data.drop_duplicates(subset=['text'], keep=False)
    data['text'] = data['text'].apply(clean_tweet)
    data['text'] = data['text'].apply(p.tokenize)

    # nlp = spacy.load("en_core_web_sm")
    # data["text"] = df["text"].apply(lambda x: [sent.text for sent in nlp(x).sents])
    # df = df.explode("Dialogue", ignore_index=True)
    
    orig = data['text'].to_list()
    data['text'] = data.text.apply(lemmatize_text)
    arr = data['text'].to_list()

    return orig, arr

def get_query(skip): # get query for 
    if (skip):
        query = pd.read_csv("./joined_query.csv")
        return query
    
    df1 = pd.read_csv("./data/data/train_tweet.csv")
    df2 = pd.read_csv("./data/data/train_tweet2.csv")
    df3 = pd.read_csv("./data/data/train_tweet3.csv")
    df4 = pd.read_csv("./data/data/train_tweet4.csv")
    df5 = pd.read_csv("./data/data/train_simple.csv")
    df6 = pd.read_csv("./data/data/train3_fake_exp.csv")
    df7 = pd.read_csv("./data/data/train4_exp.csv")

    df6 = df6[['text']]
    df7 = df7[['text']]
    df1 = df1[['text','label']]
    df2 = df2[['text','label']]
    df3 = df3[['text','label']]
    df4 = df4[['text','label']]
    df5 = df5[['text','label']]

    query = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)
    query['text'] = query['text'].apply(clean_tweet)
    query['text'] = query['text'].apply(p.tokenize)
    query['text'] = query.text.apply(lemmatize_text)

    return query

if __name__ == "__main__":
    print("Getting data and query")
    orig, arr = get_data(True)
    query = get_query(True)

    test = query['text'].to_list()
    labels = query['label'].to_list()

    

    print("Setting up BM25")
    print(len(arr))
    test = query['text'].to_list()
    search = BM25Plus(arr)


    count = 1
    start = time.time()
    print("Starting get top n")
    print("Total:",len(test))
    print("20000 to 25000")

    for idx, item in enumerate(test):
        if (count > 20000 and count <= 25000):
            label = labels[idx]
            s = str(''.join(item))
            tn = search.get_top_n(s.split(" "), orig, n=3)
            temp = str(tn)
            outer = s + "<sep>" + (str(label)+"<sep>") + (str(tn[0])+"<sep>") + (str(tn[1])+"<sep>") + (str(tn[2])+"<sep>") + ("</end>\n")
            with open(os.path.join('./test_20000_25000.txt'), 'a', encoding='utf-8') as out:
                out.write(outer)
        count += 1

    end = time.time() - start
    print("TIME TAKEN:",str(end))
    print("done")