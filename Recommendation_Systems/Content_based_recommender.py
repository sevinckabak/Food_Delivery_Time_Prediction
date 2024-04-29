#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Python/Recommendation_Systems/recommender_systems/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words='english')

# stop_words argümanı dilde yaygınca kullanılan ve ölçüm taşımayan kelimeler silmek için kullanılır.
# and, the, a gibi kelimeler.

df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')

# overview'deki boş değerleri boşlukla değiştirdik.

tfidf_matrix = tfidf.fit_transform(df['overview'])
# fit ilgili değişkenleri fit eder transform da eski değerleri fit edilmiş haliyle dönüştürür.

tfidf_matrix.shape

df["title"].shape

tfidf.get_feature_names()

tfidf.vocabulary_

tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape
# cosine_similarity der ki; bana benzerliğini hesaplamak istediğin matrisi ver.

cosine_sim[1]
# çıktı olarak matriste her filmin her film ile benzerlik skoru vardır.
# ama okumak için title bilgisi getirilmeli.


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################


indices = pd.Series(df.index, index=df["title"])
indices.index.value_counts()
# birden fazla çekilen filmlerin sonuncusunu alıp diğerlerini uçuracağız.

indices = indices[~indices.index.duplicated(keep="last")]
# dublike olan filmlerin sonuncusunu tutacak. Ve dublike olmayanları seçecek işlem gerçekleştirdik.

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]


#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]


content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
