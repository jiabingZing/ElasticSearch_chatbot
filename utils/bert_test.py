# -*- coding: utf-8 -*-
from bert_serving.client import BertClient
import jieba
bc = BertClient(port=86500, port_out=86501, show_server_config=True, timeout=1000000)
query = '人工智能终将智能！'
cut =jieba.lcut(query, cut_all=True)
print(cut)
vec = bc.encode(cut)
vec1 = bc.encode(['人工智能终将智能！'])
print(vec.shape[1])
print(vec1.tolist()[0])