


## import usfull library 
import numpy 
import pandas as pd
import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
pt=PorterStemmer()


st.header("SMS Spam Classifier")
sms=st.text_input("Enter SMS")

## this function apply lowercase, wordtokenizer , remove stopwords and after all process
 ##covert into string    

def transformer(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y.copy()
  y.clear()  
  
  for i in text:
    if i not in stopwords.words("english"):
      y.append(i)
  text=y.copy()
  y.clear()
  
  for i in text:
    y.append(pt.stem(i))
  y=' '.join(y)

  return y
if st.button('Predict'):
  
  ##vectorization

  Tfid=pickle.load(open("Tfid.pkl",'rb'))
  data=Tfid.transform([sms]).toarray()

  ##predict
  Mnb=pickle.load(open('Mnb.pkl','rb'))
  predict=Mnb.predict(data)

  ##Show Prediction

  if  predict==0:
    st.write('Not Spam')
  else:
    st.write("Spam")

     




