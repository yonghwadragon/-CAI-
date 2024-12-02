#===========================================================
# 0. 필요한 라이브러리 불러오기
#===========================================================
import os 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

import seaborn as sns

#===========================================================
# 1. CAI 계산 함수 정의
#===========================================================

def individual_cai(value, breakpoints):
    if pd.isna(value): # NAN 값 처리
        return -999
    for bp in breakpoints:
        if value <=bp['max']:
            return bp['cai']
    return 0 # 범위를 벗어난 경우 0 반환
    
# select cai
def calculate_cai(row):

    #예시: 각 오염 물질의 기준과 대응되는 CAI 범위 (대기환경 기준에 맞게 수정)
    so2_breakpoints = [{'max': 50, 'cai': 1},
                       {'max': 100, 'cai': 2},
                       {'max': 250, 'cai': 3},
                       {'max': 300, 'cai': 4}]
    co_breakpoints = [{'max': 2, 'cai': 1},
                       {'max': 9, 'cai': 2},
                       {'max': 15, 'cai': 3},
                       {'max': 30, 'cai': 4}]
    no2_breakpoints = [{'max': 0.03, 'cai': 1},
                       {'max': 0.06, 'cai': 2},
                       {'max': 0.2, 'cai': 3},
                       {'max': 0.4, 'cai': 4}]
    o3_breakpoints = [{'max': 0.03, 'cai': 1},
                       {'max': 0.09, 'cai': 2},
                       {'max': 0.15, 'cai': 3},
                       {'max': 0.2, 'cai': 4}]
    pm10_breakpoints = [{'max': 30, 'cai': 1},
                       {'max': 80, 'cai': 2},
                       {'max': 150, 'cai': 3},
                       {'max': 600, 'cai': 4}]
    pm25_breakpoints = [{'max': 15, 'cai': 1},
                       {'max': 35, 'cai': 2},
                       {'max': 75, 'cai': 3},
                       {'max': 500, 'cai': 4}]



    # 각 오염 물질의 CAI 계산
    cai_values = [
                  individual_cai(row['SO2'], so2_breakpoints),
                  individual_cai(row['CO'], co_breakpoints),
                  individual_cai(row['O3'], o3_breakpoints),
                  individual_cai(row['NO2'], no2_breakpoints),
                  individual_cai(row['PM10'], pm10_breakpoints),
                  individual_cai(row['PM25'], pm25_breakpoints)
                  ]

    # CAI 계산에서 -999는 무시하고 최댓값을 CAI로 설정
    valid_cai = [cai for cai in cai_values if cai != -999]
    return max(valid_cai) if valid_cai else -999 #유효값이 없으면 -999 반환

#===========================================================
# 2. 월별 데이터 통합
#===========================================================

# 데이터 파일들이 위치한 디렉토리 설정
data_directory =  r"data" # 실제 데이터 경로로 변경

# 엑셀 파일 목록 가져오기
file_list = [file for file in os.listdir(data_directory)
if file.endswith(".xls")]

# 데이터 프레임 생성
all_data = pd.DataFrame()

for file_name in file_list:
    # 각 엑셀 파일을 읽어서 병합
    file_path = os.path.join(data_directory, file_name)
    data = pd.read_excel(file_path)

    # 첫 번째 행 데이터 삭제
    data = data.drop(index=0).reset_index(drop=True)

    # 데이터 병합
    all_data = pd.concat([all_data, data], ignore_index=True)

# 병합된 데이터 확인
#print(all_data.head())
#print(all_data)
#print(all_data.iloc[0:10])  # 첫 10행만 출력

all_data = all_data.rename(columns= {
    '날짜': 'Date',
    'PM10': 'PM10',
    'PM2.5': 'PM25',
    '오 존': 'O3',
    '이산화질소': 'NO2',
    '일산화탄소': 'CO',
    '아황산가스': 'SO2'
})

# 숫자형으로 변환
numeric_col = ['PM10','PM25','O3','NO2','CO','SO2']
for col in numeric_col:
    all_data[col]=pd.to_numeric(all_data[col],
    errors='coerce')

# # CSV 파일로 저장
# all_data.to_csv('2023.csv', index=False, encoding='utf-8')

# 전처리
all_data['CAI'] = all_data.apply(calculate_cai, axis=1)

# 결측치 제거
all_data = all_data.dropna()

# 특징 변수 선택
features = all_data[numeric_col]

# 타켓 변수 선택
target = all_data['CAI']

# 훈련 및 테스트 데이터 분할
x_train,x_test,y_train,y_test = train_test_split(features,
                                                 target,
                                                 test_size=0.3,
                                                 random_state=42)

#===========================================================
# 3. 결정트리 모델 생성
#===========================================================
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

# 예측 수행
y_pred = model.predict(x_test)

# 성능평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred, average='weighted')
recall = recall_score(y_test,y_pred, average='weighted')
f1 = f1_score(y_test,y_pred, average='weighted') 
conf_matrix = confusion_matrix(y_test,y_pred)

# 결과 출력
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_test, y_pred))

#===========================================================
# 4. 혼동 행렬 시각화
#===========================================================

# 혼동 행렬을 데이터프레임으로 변환
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['Actual 0', 'Actual 1',
                                     'Actual 2', 'Actual 3'], # 실제
                              columns=['Predicted 0', 'Predicted 1',
                              'Predicted 2', 'Predicted 3']) # 예측

# 혼동 행렬 시각화
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_df,
            annot=True, fmt='d', cmap='Reds',
            square=True, cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#===========================================================
# 5. 결정 트리 시각화
#===========================================================

# 결정 트리 시각화
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=features.columns,
    class_names=[str(class_name) for class_name in model.classes_],
    filled=True
)
plt.show()

#===========================================================
# 6. 주어진 값으로 예측
#===========================================================

# X_new를 pd.DateFrame으로 변환하고 특성 이름을 지정
X_new = pd.DataFrame(np.array([[4.7, 1.5, 0.2, 1.4, 3.5, 5.1]]),
                     columns=['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2'])

# 예측 수행
y_pred_new = model.predict(X_new)

# 예측값 출력
print('입력값:', X_new)
print('예측값:', y_pred_new)