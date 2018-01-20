


from sklearn.datasets import load_iris
iris_dataset=load_iris()
# scikit-learn 라이브러리에서 붓꽃의 품종 데이터를 불러와 변수에 저장한다.

print("iris_dataset의 키 : {}".format(iris_dataset.keys()))
# 붓꽃의 데이터 세트의 키워드를 출력한다.
# DESCR 키워드는 데이터 세트의 간략한 설명에 대한 값을 가지고 있다.
# target_names 키워드는 분류해야 하는 붓꽃 품종의 이름 값을 가지고 있다.
# feature_names 키워드는 붓꽃 특성의 이름 값을 가지고 있다.
# data 키워드는 붓꽃의 각 특성에 해당하는 값 150개를 가지고 있다.
# target 키워드는 data에 맞는 붓꽃의 품종 값(0,1,2)을 가지고 있다.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
# scikit-learn 라이브러리의 train_test_split 함수는 전체 데이터 세트를 75%의 훈련 세트와 25%의 테스트 세트로 나누어준다.
# 기존의 데이터 세트를 그대로 나눌 경우 테스트 세트가 모두 같은 값이 될 수 있으므로, 나누기전에 데이터를 섞어준다.

import pandas as pd
import matplotlib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# scikit-learn 라이브러리의 k-NN 분류 알고리즘을 불러와 knn 객체에 저장한다.
# 훈련 데이터 셋으로부터 fit 메소드를 사용하여 모델을 만든다.

X_new=np.array([[5, 2.9, 1, 0.2]])
prediction=knn.predict(X_new)
print("예측 : {}".format(prediction))
print("예측한 타겟의 이름 : {}".format(iris_dataset['target_names'][prediction]))
# 레이블을 모르는 새로운 데이터를 정의하고, knn객체의 predict 메소드를 사용해 품종에 대한 타겟값을 예측한다.

y_pred=knn.predict(X_test)
print("테스트 세트에 대한 예측 값 :\n {}".format(y_pred))
print("테스트 세트의 정확도 : {:.2f}".format(knn.score(X_test,y_test)))
# 테스트 세트에 대한 타겟 값을 predict 메소드를 통해 예측한다.
# score 메소드를 사용하여 테스트 세트의 정확도를 계산한다.
#ㅋㅋ