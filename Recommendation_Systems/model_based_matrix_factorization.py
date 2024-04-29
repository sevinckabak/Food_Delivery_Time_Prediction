#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

!pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv("Python/Recommendation_Systems/movie.csv")
rating = pd.read_csv("Python/Recommendation_Systems/rating.csv")

df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index="userId",
                                      columns="title",
                                      values="rating")


user_movie_df.head()
user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)


##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
# matrix fonksiyonu

svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)
#rmse = hata kareler ortalaması

svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################

# modeli optimize etmek; modelin tahmin olasılığını arttırmak

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# deneyebileceğimiz farklı hiper parametreleri sözlükyapısında kaydediyoruz.
# learning katsayısı, iterasyon ve gizli faktörler.
# her birini birbiriyle eşleştirecek.

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)
# SVD fonksiyonu hiper parametreleri kullanacak. mutlak hata oranı ve hata kareler toplamı karekökü ile hatayı ölçümleyecek.
# cv=3; modeli üçe bölerek iki parçasıyla model oluşturup diğeriyle test edecek ardından diğer iki parçasıyla model oluşturup son parçayla test edecek.
# n_jobs= işlemciyi full kullan
# joblib_verbose; bana raporla çalışırken

gs.fit(data)

gs.best_score['rmse']
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse'])
# iki yıldız ile parametreleri modele koaybiliriz.

# tüm verisetini test veri setine çeviriyoruz.
data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)











