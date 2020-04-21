#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import xml.etree.cElementTree as et 


#model_dir = "./model/"
#ntcir_dir = "./CIRB010/"

#{vocab:vid}
def parse_vocab_id(model_dir):
    vocab_dic = {}
    f = open(model_dir + "vocab.all", 'r',encoding='utf-8')
    for vid,vocab in enumerate(f):
        vocab_dic[vocab.strip("\n")] = vid
    f.close
    return vocab_dic


# <id> cdn_chi_0000002 </id>
#{fid:(docName,size)}
def parse_doc_id(model_dir, ntcir_dir):
    doc_dic = {}
    all_size = 0 #prepare for okapi
    f = open(model_dir + "file-list", 'r',encoding='utf-8')
    for fid,file in enumerate(f):
       # items = list(map(lambda x:x.strip("\n"),file.split("/")))
       # path = os.path.join(ntcir_dir,items[0],items[1],items[2],items[3].lower()) 
       # path = path.replace("\\", "/")
        path = ntcir_dir + file.strip("\n")
        size = os.path.getsize(path)
        all_size += size
        items = list(map(lambda x:x.strip("\n"),file.split("/")))
        doc_dic[fid] = (items[-1].lower(),size)
    avdl = all_size/len(doc_dic) #prepare for okapi
    f.close
    return doc_dic,avdl


# {term:[total_freq,(docID,freq),(docID,freq)]}
def parse_inv(model_dir):
    f = open(model_dir + "inverted-file", 'r',encoding='utf-8')
    inv = {}
    for line in f.readlines():
        line = list(map(lambda x:int(x),line.split()))
        if len(line) == 3:
            v1,v2,cnt = line[0],line[1],line[2]
            key = (v1,v2)
            inv[key] = [cnt]
        if len(line) == 2:
            inv[key].append((line[0],line[1]))
    return inv

# query dataframe
def text_filter(sentence):
    punctuation = "，。、「」（）"
    sentence = re.sub(r'[{}]+'.format(punctuation),'',sentence)
    sentence = re.sub(r'[^\w]','',sentence)
    sentence = sentence.replace("查詢","",3).replace("相關文件內容","").replace("應","").replace("包括","").replace("應說明","")
    return sentence

def parse_query(query_file):
    dfcols = ["qid","title","question","narrative","concepts"]
    query = pd.DataFrame(columns = dfcols)
    xml_tree = et.ElementTree(file=query_file) 
    root = xml_tree.getroot()
    for i in range(len(root)):
        row = []
        obj = root.getchildren()[i].getchildren()
    
        qid = obj[0].text.strip()[-3:]
        title = obj[1].text.strip()
    
        question = text_filter(obj[2].text)
        narrative = text_filter(obj[3].text)
        concept = text_filter(obj[4].text)
    
        row = [qid,title,question,narrative,concept]
        query.loc[i] = row
    return query
