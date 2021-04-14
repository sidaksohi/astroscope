
#load and configure flask
from flask import Flask, flash, request, render_template, url_for
from flask_ngrok import run_with_ngrok

app = Flask(__name__) 
run_with_ngrok(app)

#load and configure praw
import praw
reddit = praw.Reddit(
                    client_id='7CL8VGBuxmc4cw',
                    client_secret='gHyXUkQ1jmxw4t-sGW3GTy8pfl14wA',
                    user_agent='Comment extraction (by u/secter)')

#MODEL
#imports
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import pickle as pkl
from scipy import sparse
import re
import itertools
import string
import collections
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.cluster as cluster
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report
import warnings
import joblib

#load and configure model
warnings.filterwarnings("ignore")
model = joblib.load("model.sav")
#load the vocabulary
cntizer = joblib.load("cntizer.pkl")
tfizer = joblib.load("tfizer.pkl")
#load X 
X = joblib.load("X_tfidf.pkl")
list_personality = joblib.load("list_personality.pkl")
#Download stopwords 
nltk.download('stopwords')
#Download wordnet
nltk.download('wordnet')
lemmatiser = WordNetLemmatizer()
# Remove the stop words for speed 
useless_words = stopwords.words("english")
personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                     "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]
# Remove these from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]
# Split MBTI and binarizing it
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]




#functions
def getComments(username):
  comments = []
  count = 0
  for comment in username.comments.new(limit = None):
    comments.append(comment.body)
    count += 1
  return comments, count

def getLinkKarma(username):
  linkKarma = username.link_karma
  return linkKarma

def getCommentKarma(username):
  commentKarma = username.comment_karma
  return commentKarma

def getIconImage(username):
  iconImage = username.icon_img
  return iconImage

def user_exists(name):
    try:
        username.id
    except:
        return False
    return True

def translate_personality(personality):
  # transform mbti to binary vector
  return [b_Pers[l] for l in personality]

def translate_back(personality):
  # transform binary vector to mbti personality
  s = ""
  for i, l in enumerate(personality):
    s += b_Pers_list[i][l]
  return s

def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
  list_personality = []
  list_posts = []
  len_data = len(data)
  i=0
  
  for row in data.iterrows():
      #Remove and clean comments
      posts = row[1].posts

      #Remove url links 
      temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

      #Remove Non-words - keep only words
      temp = re.sub("[^a-zA-Z]", " ", temp)

      # Remove spaces > 1
      temp = re.sub(' +', ' ', temp).lower()

      #Remove multiple letter repeating words
      temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

      #Remove stop words
      if remove_stop_words:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
      else:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
          
      #Remove MBTI personality words from posts
      if remove_mbti_profiles:
          for t in unique_type_list:
              temp = temp.replace(t,"")

      # transform mbti to binary vector
      type_labelized = translate_personality(row[1].type) #or use lab_encoder.transform([row[1].type])[0]
      list_personality.append(type_labelized)
      # the cleaned data temp is passed here
      list_posts.append(temp)

  # returns the result
  list_posts = np.array(list_posts)
  list_personality = np.array(list_personality)
  return list_posts, list_personality

def predict(my_posts):
  #Default type: ENFP
  mydata = pd.DataFrame(data={'type': ['ENFP'], 'posts': [my_posts]})

  #Process input
  my_posts, dummy  = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)

  my_X_cnt = cntizer.transform(my_posts)
  my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

  # setup parameters for xgboost
  param = {}
  param['n_estimators'] = 200
  param['max_depth'] = 2
  param['nthread'] = 8
  param['learning_rate'] = 0.2

  #XGBoost model for MBTI dataset
  result = []
  # Individually training each mbti personlity type
  for l in range(len(personality_type)):
      
      Y = list_personality[:,l]

      # split data into train and test sets
      X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

      # fit model on training data
      model = XGBClassifier(**param)
      model.fit(X_train, y_train)
      
      # make predictions for my  data
      y_pred = model.predict(my_X_tfidf)
      result.append(y_pred[0])
  return translate_back(result)




#routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/")
def index2():
    return render_template('index2.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/page_not_found")
def dev():
    return render_template('dev.html')

@app.route("/", methods =["POST"])
def results():
    if request.method == "POST":
      #get username / create redditor instance
      username = request.form.get("username")

      #return error if invalid account
      try:
          username = reddit.redditor(username)
        
          #getComments
          comments, count = getComments(username)
          my_posts = " || ".join(comments)

          #getIconImage
          iconImage = getIconImage(username)

          #getLinkKarma
          linkKarma = getLinkKarma(username)

          #getCommentKarma
          commentKarma = getCommentKarma(username)
      except:
          return render_template('index2.html')  

      #return error if no comments
      if int(count) == 0:
        return render_template('index2.html')

      #make prediction
      prediction = predict(my_posts)

      #send data to template
      return render_template('results.html', **locals())

app.run()