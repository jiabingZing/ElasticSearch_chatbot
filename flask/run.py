# -*- coding: utf-8 -*-
# @Author  : jiabing
# 想法是做个简陋的网页界面，实力不允许
from ES.es_serrch_word2vec import *
from flask import Flask
app = Flask(__name__)
app.config['DEBUG'] = True  # 开启 debug
@app.route('/')
def hello_world():
    index_name='test'
    doc_type_name='test'
    esearch = ESearch(index_name, doc_type_name)

    # 导入esearch
    #esearch.load_data_from_file(TRAIN_PATH)

    # 查询
    # name = input(output)
    context_id, context,score = esearch.query("税务注销定期定额个体工商户")
    print(context_id, context,score)
    return {
        "context_id":[id for id in context_id],
            "context": [co for co in context],
            "score": [sr for sr in score]}


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run()