#%%
import string
import pandas as pd
from PIL import Image
import streamlit as st
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stopword(x):
     return [i.rstrip('_') for i in x if i not in stopword and i not in [',', '']]

image = Image.open('./image.jpeg')
st.image(image)
st.title('Phát hiện tin giả')

model =  st.selectbox('Model you want',['LogisticRegressionCV',\
                                        'DecisionTreeClassifier'
                                        # 'MLPClassifier'
                                        ])

text =  st.text_area('Piece of news here')
#%%
# đọc tập dữ liệu từ file csv
news_df = pd.read_csv('./vn_news_223_tdlfr.csv', encoding='utf-8')
# đọc file stopwords và tổ chức dữ liệu thành một object 
stopword = open('./vietnamese-stopwords.txt', 'r',encoding='utf-8')
stopword = set([i.rstrip() for i in stopword.readlines()])
# print(stopword)
#%%
# lấy dữ liệu được nhập trên giao diện
new_df = pd.DataFrame({'text': [text], 'domain': [''], 'label': ['']})
# và nối chúng với tập dữ liệu ban đầu (tổ chức theo các fields định sắn)
new_df = pd.concat([news_df, new_df], axis=0, ignore_index=True)
# print(new_df)

#%%
# tạo regex cho tập dữ liệu
puncs = string.punctuation + '\n“”‘’'
# thay đổi những khoảng trắng trong list stopwords object thành kí hiệu _
stopword = pd.Series(list(stopword)).str.replace(' ', '_').to_list()

# xử lý dữ liệu
clean_text = new_df['text'].replace(f'[{puncs}]', ',', regex=True).\
             apply(ViTokenizer.tokenize).str.lower().\
             str.split().apply(remove_stopword)
# print(clean_text)
#%%
# tạo thêm cột mới clean_text cho tập data frame
new_df['clean_text'] = clean_text
# parse các giá trị trong mảng clean_test (của mỗi row) thành lại chuỗi
new_df['clean_text'] = new_df.apply(lambda x: ' '.join(x['clean_text']) + ' ' + x['domain'], axis=1)
# print(new_df)
#%%
# tạo ra tập dữ liệu train
train_data = TfidfVectorizer(lowercase=False).fit_transform(new_df['clean_text'])

# tập được train là tấp cả dữ liệu của tập train
x_train = train_data[:-1]
# tập test là dữ liệu được nhập sau
x_test = train_data[-1:]

# sử dụng models đc chọn để train dữ liệu
# kết quả được dự đoán sẽ dựa vào độ fit của tập test so với tập dữ liệu đã train
if st.button('Predict it news is real or fake 👈'):
    if model == 'LogisticRegressionCV':
        lg_re = LogisticRegressionCV(Cs=20, cv=5, solver='newton-cg', max_iter=10000).\
                fit(x_train, news_df.label)
        result = lg_re.predict(x_test)
        if result[0] == 1:
            st.write('fake new')
        else:
            st.write('real new')
        
    elif model == 'DecisionTreeClassifier':
        tree = DecisionTreeClassifier(random_state=42, max_features=None, max_leaf_nodes=30).\
                     fit(x_train, news_df.label)
        result = tree.predict(x_test)
        if result[0] == 1:
            st.write('fake new')
        else:
            st.write('real new')
    # else:
    #     MLPC = MLPClassifier(hidden_layer_sizes=(50), alpha=0, solver='lbfgs', max_iter=10000).\
    #            fit(x_train, news_df.label)
    #     result = MLPC.predict(x_test)
    #     if result[0] == 1:
    #         st.write('fake new')
    #     else:
    #         st.write('real new')


st.markdown('---')
st.markdown('Member: Nguyễn Bùi Duy Phong')