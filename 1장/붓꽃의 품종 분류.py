
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# 붓꽃 데이터 적재
from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)

print("X_train 크기: ", X_train.shape) # X_train 크기:  (112, 4)
print("y_train 크기: ", y_train.shape) # y_train 크기:  (112,)

print("X_test 크기: ", X_test.shape ) # X_test 크기:  (38, 4)
print("y_test 크기: ", y_test.shape) # y_test 크기:  (38,)

# X_train 데이터를 사용해서 데이터프레임을 만든다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용한다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임ㅇ르 사용하여 y_train에 따라 색으로 구분된 산점도 행렬을 만든다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker="0",
hist_kwds={"bins" : 20}, s=60, alpha=0.8, cmap=mglearn.cm3)

# k-최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# knn 객체는 훈련 데이터로 모델을 만들고, 
# 새로운 데이터 포인트에 대해 예측하는 알고리즘을 캡슐화 한 것이다.
# 또한 알고리즘이 훈련 데이터로부터 추출한 정보를 담고있다.
# KNeighborsClassifier의 경우 훈련 데이터 자체를 저장하고 있다.

# 훈련 데이터셋으로부터 모델을 만들려면 knn 객체의 fit 매서드를 사용한다.
knn.fit(X_train, y_train)

# k-최근접 이웃 알고리즘 모델 예측
X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape : ", X_new.shape)

# 예측에는 knn객체의 predict 메서드를 사용한다.
prediction = knn.predict(X_new)
print("예측 : ", prediction)
print("예측한 타깃의 이름 : ", iris_dataset["target_names"][prediction])
# k-최근접 이웃 알고리즘 모델이 꽃을 setosa 품종을 의미하는 클래스 0으로 예측한다.

# k-최근접 이웃 알고리즘 모델 평가
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값 : \n", y_pred)
print("테스트 세트의 정확도 : {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도 : {:.2f}".format(knn.score(X_test, y_test)))