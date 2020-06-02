#pip install foolnltk
import fool
test_lsting = "微信号是多少。"
words,ners = fool.analysis(test_lsting)
print(words)
print(ners)
for start, end, ner_type, ner_name in ners[0]:
    print(ner_name)


