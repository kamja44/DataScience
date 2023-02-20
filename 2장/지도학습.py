import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# forge 데이터셋 생성
X, y = mglearn.datasets.make_forge()
# 산점도
mglearn.discrete_scatter(X[:, 0],X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번쨰 특성")
print("X.shape : ", X.shape) # X.shape : (26,2) 즉, 데이터 포인트 26개와 2개의 특성

# wave 데이터셋 생성
X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,"o")
plt.ylim(-3,3)
plt.xlabel("특성")
plt.ylabel("타깃")

# 위스콘신 유방암 데이터셋
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n", cancer.keys())
print("유방암 데이터의 형태 : ", cancer.data.shape)
print("클래스별 샘플 개수: \n", 
{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
# 클래스별 샘플 개수: {'malignant': 212, 'benign': 357}
print("특성 이름 : \n", cancer.feature_names)

# 보스턴 주택가격 데이터셋
from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태 : ", boston.data.shape) # 데이터의 형태 :  (506, 13)
# 특성 공학 적용
X,y = mglearn.datasets.load_extended_boston()
print("X.shape : ", X.shape) # X.shape :  (506, 104)