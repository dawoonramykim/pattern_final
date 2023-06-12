import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 기말고사",
)

st.header("Linear Regression with MSE")
st.markdown("")
st.markdown("선형회귀(Linear Regression)는 y=f(x)+ε의 형태로 입력 변수 x와 출력 변수 y 사이의 선형 상관 관계를 모델링하는 분석 기법이다. 입력 변수를 통해 출력 변수를 예측하거나, 두 변수 사이의 관계를 규명할 때 사용된다.")
st.markdown("특별히 평균제곱오차(Mean Squared Error)를 같이 실시하여 실제 값과 예측의 오차를 나타타어 예측 모델이 얼마나 정확한지 평가할 수 있다.")

random = st.sidebar.number_input('0부터 100 중 랜덤 값을 선택하십시오.', 0, 100, 42)
st.write("The current random value is ", random)

np.random.seed(random)
x=np.random.rand(100,1)
y=2*x+1+0.1*np.random.randn(100,1)


model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)

mse=np.mean((y-y_pred)**2)
print(f"Mean Squared Error:{mse}")

plt.scatter(x,y,label='Data')
plt.plot(x,y_pred,color='red',label='Linear Regression')
plt.title('Linear Regresion with MSE')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()
st.pyplot(plt)
st.write("Mean Squred Error : ",mse)