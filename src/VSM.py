#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import argparse
from utiles import *
#from MAP import *
parser = argparse.ArgumentParser(description='vsm')
parser.add_argument("-r", action="store_true", help="turn on feedback")
parser.add_argument("-i", default="../queries/query-test.xml", type=str, help="query file path")
parser.add_argument("-o", default="../ranked-list.csv", type=str, help="output ranked file .csv")
parser.add_argument("-m", default="../model/", type=str, help="model dir")
parser.add_argument("-d", default="../CIRB010/", type=str, help="docs dir")
args = parser.parse_args()


def vsm(qidx, weight=[], fb=False):
    #計算query中的詞頻 qt={(v1,v2):qf,....}
    search = query.iloc[qidx]["title"]+query.iloc[qidx]["question"]+query.iloc[qidx]["narrative"]+query.iloc[qidx]["concepts"]               
    qt = {}
    m = 0
    for i in range(len(search)-1):
        term = (vocab_to_id[search[i]], vocab_to_id[search[i+1]])
        if term not in inv:
            continue
        if fb == True and term not in qt:
            qt[term] = weight[qidx][m]
            m += 1
        else:
            if term not in qt:
                qt[term] = 1
            else:
                qt[term] += 1
    #計算一篇文章的BM25得分
    doc2vec = {}
    query2vec = [0]*len(qt)
    qv_idx = 0
    for term, qf in qt.items():
        k1 = 1.0
        k3 = 1.5
        df = inv[term][0]
        IDF = math.log((N - df + 0.5) / (df + 0.5))
        QTF =  ((k3 + 1) * qf) / (k3 + qf)
        for post in inv[term][1:]:
            tf = post[1]
            doc_id = post[0]
            doc_name = file_to_id[doc_id][0]
            dl = file_to_id[doc_id][1]
            TF_norm =  ((k1 + 1) * tf) / (k1 * (0.25 + 0.75 * dl / avdl) + tf)
            score = IDF*TF_norm*QTF
            if doc_name not in doc2vec:
                doc2vec[doc_name] = [0]*len(qt)
            doc2vec[doc_name][qv_idx] = score
        query2vec[qv_idx] = qf
        qv_idx += 1
    return query2vec,doc2vec


def rank(doc2vec):
    doc2score = {}
    for rel_doc,rel_score in doc2vec.items():
        doc2score[rel_doc] = sum(rel_score)
    rank = pd.DataFrame(pd.Series(doc2score),columns=["score"])
    sort = rank.sort_values(by="score",ascending=False)
    return sort


def topk(sort,k=100):
    d = sort.index[0]
    for i in range(1,k):
        d += ' '+ sort.index[i]
    return d

def feedback(result,query2vec,doc2vec):
    thr = 40
    alpha, beta, gamma = 0.9, 0.3, 0.2
    q_m = []
    for i in range(len(result)):
        fetched = result.iloc[i][1].split()
        D_r = fetched[:thr]
        D_n = fetched[-thr:]
        origin = np.array(query2vec[i])
        docs = doc2vec[i]
        d_r = np.zeros(len(origin))
        d_n = np.zeros(len(origin))
        for dr in D_r:
            if docs.__contains__(dr):
                d_r += np.array(docs[dr])
                #d_r = [x+y for x,y in zip(d_r,docs[dr])]
        d_r = d_r/len(D_r)
        #d_r = [ x/len(D_r) for x in d_r]
        for dn in D_n:
            if docs.__contains__(dn):
                d_n += np.array(docs[dn])
                #d_n = [x+y for x,y in zip(d_n,docs[dn])]
        d_n = d_n/len(D_n)
        #d_n = [ x/len(D_n) for x in d_n]
        q_m.append(alpha*origin + beta*d_r - gamma*d_n)
    return q_m


def result(query,weight,feedback):
    result = {}
    doc2vec = []
    query2vec = []
    for qidx in range(len(query)):
        q2v,d2v = vsm(qidx,weight,feedback)
        query2vec.append(q2v)
        doc2vec.append(d2v)
        sort = rank(d2v)
        q = query.iloc[qidx]['qid'] 
        result[q] = topk(sort, 100)
    result = pd.DataFrame(pd.Series(result),columns=["retrieved_docs"])
    result = result.reset_index().rename(columns = {"index":"query_id"})
    return result,query2vec,doc2vec


def adjust(ad_rank, ad_q2v, ad_d2v):
    new_weight = feedback(ad_rank,ad_q2v,ad_d2v)
    predict,q2v,d2v = result(query = query, weight = new_weight,feedback = True)
    return predict,q2v,d2v    


def reFB(times,ini_rank,ini_q2v,ini_doc2v):
    for i in range(times):
        if i == 0:
            predict,q2v,d2v = ini_rank,ini_q2v,ini_doc2v 
        else:
            predict,q2v,d2v = adjust(predict, q2v, d2v)
            #predict.to_csv("predict.csv",index = 0)
            #print("MAP@100 = ",MAP())
    return predict


model_dir = args.m
ntcir_dir = args.d
query_file = args.i

vocab_to_id = parse_vocab_id(model_dir)
file_to_id,avdl = parse_doc_id(model_dir,ntcir_dir)
inv = parse_inv(model_dir)
N = len(file_to_id)

query = parse_query(query_file)

rel_fb = args.r

ini_rank,ini_q2v,ini_doc2v = result(query = query, weight = [], feedback = False)
if rel_fb == True:
    predict = reFB(2,ini_rank,ini_q2v,ini_doc2v)
else:
    predict = ini_rank


predict.to_csv(args.o,index = 0)

print('Finish running\nDumping ranked-list under ./')

