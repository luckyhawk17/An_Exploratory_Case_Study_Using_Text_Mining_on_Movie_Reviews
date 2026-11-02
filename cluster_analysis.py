import glob
import pandas as pd
import scipy.stats as ss
import morphological_analysis as mp
import time
import numpy as np
from sklearn.cluster import KMeans
import MeCab
import re
from sklearn.feature_extraction.text import TfidfVectorizer

start = time.time()

min_df = 200  # 単語の最低度数（整数値）
max_df = 0.4 # 単語の最大登場割合（1以下の正の小数）

n_clusters = 8 # クラスター数

top = 10

# ストップワード
stopwords_ad = ['凄い', '良い', '無い'] # 形容詞
stopwords_pr = ['機', '気', 'こと', '事', '人', 'ため', '為', '度', '時', '年', 'ところ', '何', '中', '方', '前', 'もの'] # 名詞

REVIEWS_PATH = "../reviews/"
all_files = glob.glob("{}*.csv".format(REVIEWS_PATH))
number_of_all_titles = len(all_files)

reviews = []
for file in all_files:
    opened_f = open(file, encoding="utf-8_sig")
    reviews.append(pd.read_csv(opened_f))
    opened_f.close()

df = pd.concat(reviews, sort = False)
df = df.reset_index(drop = True)
df.index.name = "総合インデックス"
df = df.dropna(how='all')
n = len(df)


## 形態素解析
title = df.iloc[1, 1]
indices = df['インデックス']
indices_list = list(indices)

tagger = MeCab.Tagger('-d "C:/Users/*****/AppData/Roaming/Python/Python313/site-packages/unidic/dicdir"')

morphoed_review_list = []
corpus = []
all_review = ""
for index in indices_list:
    
    review = df[df['インデックス'] == index]
    review = review['レビュー']
    review = "".join(map(str, review))
    all_review += review

    parsed = tagger.parse(review)

    lines = parsed.split('\n')
    items = (re.split('[\t,、".．「」!！-－・?？【】()（）…/~～_]', line) for line in lines)
    
    morphoed_review = []
    corp = ""
    for item in items:            
        word = []

        if item[0] not in ('EOS', '') :
            match item[1]:
                case '名詞':
                    if item[0] in stopwords_pr:
                        continue
                    else:
                        word.append(item[0])
                        word.append(item[1])
                        morphoed_review.append(word)
                        corp += item[0] + " "                
                
                case '形容詞':
                    if item[8] in stopwords_ad:
                       continue
                    else:
                       word.append(item[8])
                       word.append(item[1]) 
                       morphoed_review.append(word)
                       corp += item[8] + " "

    morphoed_review_list.append(morphoed_review)
    corp = corp[:-1]
    corpus.append(corp)    

morphoed_all_review = []
for morphoed_review in morphoed_review_list:
    morphoed_all_review += morphoed_review
      
all_freq_list = mp.morphological_analysis.generate_freq_list(morphoed_all_review)
all_freq_list = mp.morphological_analysis.bubble_sort(all_freq_list) 

all_word_list = []
for word in all_freq_list:
    all_word_list.append(word[0])

all_word_list_l = []
for w in all_word_list:
    all_word_list_l.append(w[0])


## TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, lowercase=False, norm='l2') #L2ノルムで正規化
X =vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out() # get_feature_names()の新バージョン
feature_names_df = pd.DataFrame(feature_names, columns=['特徴量'])
feature_names_df.to_csv("feature_names_df.csv", index=True, encoding="utf-8_sig")

tf_idf_matrix = pd.DataFrame(X.todense(), columns=feature_names)
tf_idf_matrix.to_csv("tf_idf.csv", index=True, encoding="utf-8_sig")

number_of_lines_in_tf_idf_matrix = len(tf_idf_matrix)
number_of_columns_in_tf_idf_matrix =  len(tf_idf_matrix.columns)

print("特徴量ラベル:", feature_names)


## クラスター分類
km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, max_iter=300, tol=0.0001)
clusters = km.fit_predict(tf_idf_matrix)
centroids = km.cluster_centers_
centroids_df = pd.DataFrame(centroids)
centroids_df.to_csv("centroids_df.csv", index=True, encoding="utf-8_sig")

tf_idf_matrix.insert(loc=0, column='クラスター', value=clusters)
tf_idf_matrix.to_csv("tf_idf.csv", index=True, encoding="utf-8_sig")
df.insert(loc=0, column='クラスター', value=clusters)
df.to_csv("df.csv", index=True, encoding="utf-8_sig")

# 重心に最も近いサンプルを抽出する
cluster_number = 0
for target_point in centroids:
    points = tf_idf_matrix[tf_idf_matrix['クラスター'] == cluster_number]
    original_index = points.index
    original_index = original_index.to_numpy()
    points_array = points.to_numpy()
    points_array = np.delete(points_array, 0, axis=1)
    diff = points_array - target_point
    distances = np.linalg.norm(diff, axis=1)
    dist_df = pd.DataFrame(distances, columns=['距離'])
    dist_df.insert(loc=0, column='元のインデックス', value=original_index, )
    dist_df = dist_df.sort_values('距離')
    dist_df.to_csv("dist_df_{0}.csv".format(cluster_number), index=False, encoding="utf-8_sig")
    
    cluster_number += 1

## クラスターによる対応分析用のクロス表を作成
cluster_all_review = df['レビュー']
cluster_all_review = "".join(cluster_all_review)

# 各クラスターの形態素解析
all_cluster_morphoed_reviews = []
number_of_reviews_in_the_cluster_list = []
number = 0
cluster_data_list = []
for cluster in range(0, n_clusters, 1):

    letters = df[df['クラスター'] == cluster]
    n_samples = len(letters)
    letters = letters['レビュー']    
    letters = "".join(map(str, letters))
    n_letters = len(letters)
    ave = n_letters/n_samples
    print("クラスター:{0}, サンプル数:{1}, 平均文字数:{2:.2f}".format(number, n_samples, ave))
    cluster_data = []
    cluster_data.append(number)
    cluster_data.append(n_samples)
    cluster_data.append(ave)
    cluster_data_list.append(cluster_data)

    number += 1

    review = df[df['クラスター'] == cluster]
    cluster_indices = list(review['インデックス'])

    cluster_morphoed_reviews = []
    number_of_reviews_in_the_cluster = len(review)
    number_of_reviews_in_the_cluster_list.append(number_of_reviews_in_the_cluster)
    for index in cluster_indices:
        review = df[df['インデックス'] == index] 
        review = review['レビュー']
        review = "".join(map(str, review))
        
        tagger = MeCab.Tagger('-d "C:/Users/*****/AppData/Roaming/Python/Python313/site-packages/unidic/dicdir"')
        parsed = tagger.parse(review)

        lines = parsed.split('\n')
        items = (re.split('[\t,、".．「」!！-－・?？【】()（）…/~～_]', line) for line in lines)

        morphoed_review = []
        for item in items:            
            word = []

            if item[0] not in ('EOS', '') :
                match item[1]:
                    case '名詞':
                        if item[0] in stopwords_pr:
                            continue
                        else:
                            word.append(item[0])
                            word.append(item[1])
                            morphoed_review.append(word)
                    
                    case '形容詞':
                        if item[8] in stopwords_ad:
                            continue
                        else:
                            word.append(item[8])
                            word.append(item[1]) 
                            morphoed_review.append(word)

        cluster_morphoed_reviews.append(morphoed_review)

    all_cluster_morphoed_reviews.append(cluster_morphoed_reviews)

cluster_data_df_columns = ['クラスター', 'サンプル数', '平均文字数']
cluster_data_df = pd.DataFrame(cluster_data_list, columns=cluster_data_df_columns)
cluster_data_df.to_csv("cluster_data.csv", encoding="utf-8_sig")
cluster_word_freq_list = []
for cluster_morphoed_reviews in all_cluster_morphoed_reviews:
    freq = []
    for word in all_word_list:   
        number_of_the_word = 0
        for review_in_the_cluster in cluster_morphoed_reviews:
            if word in review_in_the_cluster:
                number_of_the_word += 1

        freq.append(number_of_the_word)

    cluster_word_freq_list.append(freq)


cluster_list = []
for i in range(0, n_clusters, 1):
    cluster_list.append(i)

cluster_word_df = pd.DataFrame(data=cluster_word_freq_list, columns=all_word_list_l)
cluster_word_df.insert(loc=0, column="クラスター", value=cluster_list)
cluster_word_df.to_csv("cluster_word_list.csv", encoding="utf-8_sig")

data = pd.DataFrame()
data["クラスター"] = cluster_list
chi2_table = pd.DataFrame()
for i in range(1, top + 1, 1):
    column = cluster_word_df.iloc[:, i]

    crossed = []
    for j in range(0, len(cluster_list), 1):       
        one = column[j]
        zero = number_of_reviews_in_the_cluster_list[j] - one
        row = []
        row.append(int(zero))
        row.append(int(one))
        crossed.append(row)

    chi2_table.insert(loc=0, value=crossed, column=column.name)
    x2, p, dof, expected = ss.chi2_contingency(crossed, correction=True)
    
    if p < 0.001:
        data[column.name] = column
        
        if p < 0.0001:
            data.rename(columns={column.name:column.name+'**'}, inplace=True)
        elif p < 0.00001:
            data.rename(columns={column.name:column.name+'***'}, inplace=True)
        else:
            data.rename(columns={column.name:column.name+'*'}, inplace=True) 

    crossed.clear()

chi2_table.to_csv("chi2_table.csv",  encoding="utf-8_sig")

# クロス表のcsvファイル
data.to_csv("data.csv", index=False, encoding="shift-jis") 

print("全文字数：{0}".format(len(all_review)))
print("投稿者数：{0}".format(len(indices_list)))
print("TF-IDF行列の行数:{0}".format(len(tf_idf_matrix)))
print("TF-IDF行列の列数:{0}".format(len(tf_idf_matrix.columns)))

end = time.time()
elapsed = end - start
hour = elapsed // 3600
minute = (elapsed % 3600) // 60
second = (elapsed % 3600 % 60)
elapsed_text = "処理時間：%d時間%d分%d秒" % (hour, minute, second)
print(elapsed_text)

input()