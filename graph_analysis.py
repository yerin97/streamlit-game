import sacrebleu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

def word_len(data_path):
  input = open(data_path,"r")
  rawText = input.read().strip()
  lines = rawText.split("\n")
  word_data = []
  word_len = []
  sen_data = []
  for sen in lines:
    words = sen.split(' ')
    sen_data.append(sen)
    word_data.append(words)
    word_len.append(len(words))
  return word_len, word_data, sen_data

def create_graph(source_data,target_data,pred_data):
  source_sen_len, source_words, source_sen = word_len(source_data)
  true_sen_len, true_words, true_sen = word_len(target_data)
  pred_sen_len, pred_words, pred_sen = word_len(pred_data)
  score_len = []

  for i in range(len(true_words)):
    score = sacrebleu.sentence_bleu(true_sen[i], [pred_sen[i]],  smooth_method='exp').score
    score_len.append([score,source_sen_len[i]-pred_sen_len[i]])

  df = pd.DataFrame(score_len).rename(columns={0:'score',1:'length difference'})

  p = sns.regplot(x='length difference', y='score', data= df)
  p.set(xlabel = "Sentence Length", ylabel = "BLEU Score", title = 'Relationship between sentence length difference and BLEU Score')
  plt.savefig("BLEU_score_graph.png")
  plt.show()
       
if __name__ == '__main__':
    source = sys.argv[1]
    target = sys.argv[2]
    pred = sys.argv[3]
    create_graph(source,target,pred)
