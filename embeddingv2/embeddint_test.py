from text2vec import SentenceModel
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

model = SentenceModel('text2vec-base-chinese')
embeddings = model.encode(sentences)
print(embeddings)
