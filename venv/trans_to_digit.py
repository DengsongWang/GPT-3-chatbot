import pandas as pd
import numpy as np
import openai
openai.api_key = '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'
datafile_path = "answer.csv"  # for your convenience, we precomputed the embeddings
df = pd.read_csv(datafile_path)
def get_embedding(text, model="text-similarity-davinci-001"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


df['question'] = df.question.apply(lambda x: get_embedding(x, model='text-similarity-babbage-001'))
df['answer'] = df.answer.apply(lambda x: get_embedding(x, model='text-search-babbage-doc-001'))
df.to_csv('digit.csv', index=False)