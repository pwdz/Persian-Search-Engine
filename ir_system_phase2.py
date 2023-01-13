from __future__ import unicode_literals
from cmath import log10, sqrt
import os
import sys
from configs import *
from hazm import *
import json
from parsivar import FindStems
# from configs import Stop_words, Punctuations, Writing_marks
import tqdm
from heapq import heappop, heappush, heapify
 
IR_DATA_PATH = "./IR_data_news_12k.json" 
INDEX_FILE_PATH = "./inverted_index.json" 
CHAMPIONS_FILE_PATH = "./champions_list.json" 

normalizer = Normalizer()
lemmatizer = Lemmatizer()
my_stemmer = FindStems()

inverted_index = {}
champions_list = {}
r = 300
news_data = None


def read_json(file_path):
    global news_data
    f = open(file_path)
    news_data = json.load(f)
    f.close()


def remove_punctuations(txt):
    for p in Punctuations:
        if p in txt:
            txt = txt.replace(p, "")
    return txt

 
def preprocess_txt(txt, with_puncremove=True, with_normalization=True, with_stop_word=False, with_lemmatization=True):
    # remove puncs
    if with_puncremove:
        txt = remove_punctuations(txt)          

    if with_normalization:
        txt = normalizer.normalize(txt)
    tokens = word_tokenize(txt)

    i = 0
    while i < len(tokens):
        # remove stop words & wrting marks
        if tokens[i] in Writing_marks:
            tokens.pop(i)
            continue
        
        if tokens[i] in Stop_words and not with_stop_word:
            tokens.pop(i)
            continue

        if with_lemmatization:
            tokens[i] = lemmatizer.lemmatize(tokens[i])
        i += 1
    
    return tokens


def build_inverted_index(inverted_idx, tokens, doc_id):
    for i in range(len(tokens)):

        if tokens[i] not in inverted_idx:
            inverted_idx[tokens[i]] = {'df': 0, 'doc_tf': {}}
        if doc_id not in inverted_idx[tokens[i]]['doc_tf']:
            inverted_idx[tokens[i]]['doc_tf'][doc_id] = 1
            inverted_idx[tokens[i]]['df'] += 1
        else:
            inverted_idx[tokens[i]]['doc_tf'][doc_id] += 1


def build_champions_list():
    for term in tqdm.tqdm(inverted_index):
        sorted_a = list(sorted(inverted_index[term]["doc_tf"].items(), reverse=True, key=lambda item: item[1]))[:r]
        term_r_sorted_doc_tf = [pair[0] for pair in sorted_a]
        champions_list[term] = term_r_sorted_doc_tf
    
        
def process_data(data, with_stop_word=False, doc_length=0, with_lemmatization=True):
    i = 0
    if doc_length == 0:
        doc_length = len(data)
    print("creating indexes...")

    for doc_id in tqdm.tqdm(data):
        i+=1
        article = data[doc_id]
        
        tokens = preprocess_txt(article['content'], with_stop_word= with_stop_word, with_lemmatization=with_lemmatization) 
        build_inverted_index(inverted_index, tokens, doc_id)
        if int(doc_id) >= doc_length:
            break

    print("creating indexes DONE")

    print("creating champions list...")
    build_champions_list()
    print("creating champions list DONE")
    
    
def save_data(path, data):    
    f = open(path, "w")
    json.dump(data, f)
    f.close()
    print(f"{path} saved.")

    
def load_data(path):

    print(f"loading {path} ...")
    f = open(path, "rb")
    data = json.loads(f.read())
    f.close()
    print(f"loading {path} DONE")
    return data


def calculate_tfidf(tf, df, N):
    return (1+log10(tf)) * log10(N/df)
     

def calculate_score(scores, doc_id, q_term, q_vector):
    if doc_id not in scores:
        scores[doc_id] = 0
    
    df = inverted_index[q_term]['df']
    N = len(news_data)
    doc_tf = inverted_index[q_term]['doc_tf'][doc_id]
    W_td = calculate_tfidf(doc_tf, df, N)

    query_tf = q_vector[q_term]['doc_tf']['query']
    W_q = calculate_tfidf(query_tf, df, N)
            
    scores[doc_id] += W_q * W_td
    return W_td
        
def calculate_length(lengths, doc_id, w_td):
    lengths[doc_id] += w_td**2
        
    
def calculate_similarity(query_tokens, goal_docs, q_vector):
    scores = {}
    lengths = {}
    for q_term in query_tokens:
        for doc_id in goal_docs:
            if doc_id not in scores:
                scores[doc_id] = 0
            if doc_id not in lengths:
                lengths[doc_id] = 0

            if doc_id in inverted_index[q_term]['doc_tf']:
                w_td = calculate_score(scores, doc_id, q_term, q_vector)
                calculate_length(lengths, doc_id, w_td)

    for doc_id in scores:
        scores[doc_id] /= sqrt(lengths[doc_id])
    return scores
    
    

def search_query(query):
    query_tokens = preprocess_txt(query)
    vector = {}
    build_inverted_index(vector, query_tokens, 'query')
   
    goal_docs = set()
    for q_term in query_tokens:
        goal_docs |= set(doc_id for doc_id in champions_list[q_term])

    scores = calculate_similarity(query_tokens, goal_docs, vector)
    return scores


def show_results(scores, top_k):
    pass
    print("\n[[[[[[[Results]]]]]]]:")
    print(f"Total found results: {len(scores)}")
    heap = []
    heapify(heap)   
    for key in scores:
        heappush(heap, (-1 * scores[key].real, key))
    
    for i in range(top_k):
        if i >= len(scores):
            break
        print('=============================================')
        score, doc_id = heappop(heap)
        score *= -1
        print(f"{i+1}) [doc_id: {doc_id}], [Score: {score.real}], [Title: {news_data[doc_id]['title']}]")
        print("[Link:", news_data[doc_id]['url'],"]")
    
                     
if __name__ == "__main__":
    read_json(IR_DATA_PATH)    
    
    if "--createIndex" in sys.argv or not os.path.isfile(INDEX_FILE_PATH):
        process_data(news_data)
        save_data(INDEX_FILE_PATH, inverted_index)
        save_data(CHAMPIONS_FILE_PATH, champions_list)
    else:    
        inverted_index = load_data(INDEX_FILE_PATH)
        champions_list = load_data(CHAMPIONS_FILE_PATH)
        
    print("Enter query:")
    query = input()
   
    scores = search_query(query)
    show_results(scores, 20)