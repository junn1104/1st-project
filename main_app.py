import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
df = pd.read_csv("diabetes_prediction_dataset.csv")


# 범주형 데이터 인코딩
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df['smoking_history'] = LabelEncoder().fit_transform(df['smoking_history'])

# 피처/타겟 분리
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#비율은 조절 가능함 바꿔도 됨!

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 성능 출력
'''
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
'''


# 모델 저장
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(X.columns.tolist(), "diabetes_model_features.pkl")


from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# 저장된 모델 불러오기 (한 번만 실행됨)
model = joblib.load("diabetes_model.pkl")

smoking_map = {
    "never": 3,
    "No Info": 2,
    "former": 0,
    "current": 1,
    "ever": 4,
    "not current": 5
}

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # 입력값 수집
            gender = request.form.get("gender")
            age = float(request.form.get("age"))
            hypertension = int(request.form.get("hypertension"))
            heart_disease = int(request.form.get("heart_disease"))
            smoking_history = request.form.get("smoking_history")
            bmi = float(request.form.get("bmi"))
            hba1c = float(request.form.get("HbA1C_level"))
            glucose = float(request.form.get("blood_glucose_level"))

            gender_encoded = 1 if gender == "Male" else 0
            smoking_encoded = smoking_map.get(smoking_history, 2)

            # 예측을 위한 배열 구성
            input_data = np.array([[gender_encoded, age, hypertension, heart_disease,
                                    smoking_encoded, bmi, hba1c, glucose]])

            # 예측 수행
            prediction = model.predict(input_data)[0]

            result = "High risk of diabetes" if prediction == 1 else "Low risk of diabetes"

            return f"<h3>예측 결과: {result}</h3><br><a href='/'>다시 입력하기</a>"

        except Exception as e:
            return f"<h3>오류 발생: {str(e)}</h3><br><a href='/'>다시 시도하기</a>"

    else:
        return render_template("main.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5051)
