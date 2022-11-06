# Dengsong Wang Week 1 and week 2 Deliverables
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

datafile_path = "digit.csv"
datafile_path2 = "answer.csv"
df = pd.read_csv(datafile_path)
origin_file = pd.read_csv(datafile_path2)
df["question"] = df.question.apply(eval).apply(np.array)

def search_reviews(df, product_description):
  embedding = get_embedding(
    product_description,
    engine="text-search-babbage-query-001"
  )
  df["similarities"] = df.question.apply(lambda x: cosine_similarity(x, embedding))

  res = (
    df.sort_values("similarities", ascending=False)
  )
  return ("Answer 2: " + origin_file.at[res.index[0], 'answer']+"\n\n"+"Answer 3: " + origin_file.at[res.index[1], 'answer']+"\n\n"+"You might want to ask: " + origin_file.at[res.index[0], 'question'])

def result():
  openai.api_key = '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'
  while True:
    user_input = input("Human: ")
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=user_input,
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6
    )
    gpt3_answer = response["choices"][0]["text"].replace("\n", "") # GPT3 answer
    other_answers_and_questions = search_reviews(df, user_input) # answer 2, 3 and might ask question
    print("GTP-3 answer: "+gpt3_answer+"\n")
    print(other_answers_and_questions)

if __name__ == '__main__':
    result()
