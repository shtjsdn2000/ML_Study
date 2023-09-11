import matplotlib.pyplot as plt
import pandas as pd #pandas 데이터르 불러오느 라이브러리 주로 엑셀 데이터를 불러옴
from sklearn.model_selection import train_test_split #데이터 분리
from sklearn.preprocessing import StandardScaler #데이터를 표준화점수
from sklearn.neighbors import KNeighborsClassifier # 주변에 있는 데이터를 활용 확률표시
import numpy as np
#데이터를 불러옴 그리고 가공
fp = pd.read_excel('fp.xlsx')
ep = pd.read_excel('ep.xlsx')
fp_test = fp.iloc[5:, 13:]
fp_test = fp_test.reset_index(drop=True)
fp_test.index = pd.RangeIndex(start=1, stop=len(fp_test)+1)
fp_test.columns = ["무연탄","유류","LNG"]

ep_test = ep.iloc[3:, 3:-1]
ep_test = ep_test.reset_index(drop=True)
ep_test.index = pd.RangeIndex(start=1, stop=len(ep_test)+1)
ep_test.columns = ["총합"]

# fish.head()
#fp = fp.filter(["무연탄","유류","LNG"])
print(fp_test)
fp_test.to_excel(excel_writer='fp_test.xlsx')
ep_test.to_excel(excel_writer='ep_test.xlsx')

ep.head()
plt.plot(ep_test)
plt.show()
plt.plot(fp_test)
plt.show()
#그래프로 경향성을 봤을때 영향을 주는 LNG,유류,무연탄의 연료비 단가와
#타겟데이터 총합만 남긴 후 데이터의 갯수를 맞춰줌
#여기까지 데이터 가공 과정
print("pandas:",pd.unique(ep_test['총합']))
energy_input = fp_test[["무연탄","유류","LNG"]].to_numpy() # #fp = fp.filter(["무연탄","유류","LNG"])
print("입력되는 연료 종류 :",energy_input[:3])
# fish_target = fish['Species'].to_numpy()
energy_target = ep_test["총합"].to_numpy()

# 데이터분리
train_input, test_input, train_target, test_target = train_test_split(energy_input,energy_target, random_state=42)
#데이터 표준화
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

print("데이터의 크기", np.shape(train_target))
# #앞에서 10개만 자르겠음
t_target = train_target[0:10]
# #벡터식을 사용해 for문을 안써도 됨 // 벡터식임
print("총합",np.sum(t_target=="총합"))
# z = a * 무게 + b * 길이 + c * 대각선 + d * 높이 + e * 두께 + f //1차식
# z > 1 을 만들기 위해 abcdef를 조정해 1 을 넘기게 해줌 (음수값을 가질 수 있음)
# ex) z = 도미 // abcdef = 다른 물고기 종류
#도미데이터의 특징을 반영해 도미일경우 z 을 1보다 크게만들지만 다른경우는 1보다 작게 되게끔 알아서 조정해줌
#--> 그것이 로지스틱 회귀
#sigmoid(1/1+e^-z) 함수일 경우 z에 무한대 넣을경우 --> 1 // 마이너스 무한대는 --> 0 에 가까워짐


#로지스틱 회귀 (이진분류)
#Bream 과 Smelt의 갯수를 표시해라!
from sklearn.linear_model import LinearRegression
#fit을 통해 계산되는것? --> abcdef값 -->왜계산이 될까? 값을 넣었으니깐!
#lr.fit(train_bream_smelt, target_bream_smelt) #<--이때 abcde 값을 정함

lr = LinearRegression()
lr.fit(train_scaled, train_target)

# 학습 정확도, 테스트 정확도 출력
print("학습 정확도 :", lr.score(train_scaled, train_target))
print("테스트 정확도 :", lr.score(test_scaled, test_target))
print("정확도 차이 : ",lr.score(train_scaled, train_target)-lr.score(test_scaled, test_target))

#2023_03_29 과제
#Regression : 학습데이터와 - Target 의 상관관계가 있어야함
#두개의 엑셀 파일을 가지고 회귀모델을 하나 만들어볼것
#학습정확도를 얼만큼이상올릴것
#week3에 썼던 리그레이션을 활용할것 (3차정도 무난 // 4,5까지도 볼 수 있음)
#데이터 : 연료가격과 전기가격과의 회귀모델
#대한민국 : 자연에너지 10% / 1. 석탄화력발전소가 제일 많이 사용됨 / 2.원자력 ...석탄과 비슷 비슷/
#/3.석유 /4.LPG 우리나라는 마지막에 사용되는 에너지원의 가격을 따라감 여기선 lpg
#주어진 데이터 종류 : 원자력/ 석유 / lpg
#159행 3열
#아웃풋(타겟) 전기 가격
#0.데이터를받음
#1.데이터를 그려봄 (plot를 활용)
#1-1 그래프를 보고 전기가격과 상관있는지를 판단
#2.3가자의 데이터를 모두 사용해야 될까? 고민해야됨(원자력같은경우는 영향을 거의안줌)
#3.데이터수가 많지않음 따라서 값을 조점하면서 다양한시도를 해봐야함
