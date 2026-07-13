#Orange3 
                        AUC	CA	F1	Precision	Recall	MCC
Logistic Regression-200	0.81	0.65	0.60	0.61	0.65	0.44
Neural Network-SGD	0.80	0.64	0.60	0.60	0.64	0.43
Random Forest-70	0.77	0.61	0.59	0.58	0.61	0.38
Naive Bayes (2)	0.77	0.61	0.59	0.58	0.61	0.38
kNN-7	0.74	0.58	0.56	0.55	0.58	0.33
SVM-Linear	0.56	0.42	0.40	0.40	0.42	0.06
Logistic Regression-L1	0.81	0.65	0.60	0.62	0.65	0.44
Logistic Regression-L1 (50)	0.81	0.65	0.60	0.61	0.65	0.44
Logistic Regression-L1 (200)	0.81	0.65	0.60	0.61	0.65	0.44
Logistic Regression-1000	0.81	0.65	0.60	0.61	0.65	0.44
Stack	0.80	0.64	0.58	0.58	0.64	0.42

#python
#LogisticRegression (scikit-learn)
#- 정규화: Ridge (L2)
#- C: 1.0 (기본값)
#- max_iter: 2000
#- multi_class: auto (기본값 → 다중클래스 자동 처리)
#- solver: lbfgs (기본값)

AUC	CA	F1	Precision	Recall	MCC
0.8081	0.6406	0.5963	0.5966	0.6406	0.4243

#결과 해석
#orang3와 python code로 예측을 수행했을때 차이는 유의미 하지 않았다.
#그 이유는 데이터셋의 12개의 특성이 예측결과에 큰 변화를 이끌어내지 못했다.
#(파라미터값 변경 등 수행하면서) 

파이썬은 디테일한 설정값 수정이 가능하였고
orange3는 직관적인 위젯연결 GUI 인터페이스가 제공된다.

타인이 수집 및 가공한 데이터셋을 우리에 맞게 활용하는것은 생각한거 보다 다각전 접근과 신중한 접근이 필요하다.
