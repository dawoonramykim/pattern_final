import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error




st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 기말고사",
)

st.header("Linear Regression")
st.markdown("")
st.markdown("선형회귀(Linear Regression)는 y=f(x)+ε의 형태로 입력 변수 x와 출력 변수 y 사이의 선형 상관 관계를 모델링하는 분석 기법이다. 입력 변수를 통해 출력 변수를 예측하거나, 두 변수 사이의 관계를 규명할 때 사용된다.")

random = st.sidebar.slider('0부터 100 중 랜덤 값을 선택하십시오.', 0, 100, 42)
st.write("The current random value is ", random)

np.random.seed(random)
x=np.random.rand(100,1)
y=2*x+1+0.1*np.random.randn(100,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

plt.scatter(x,y,color='blue',label='Actual Data')
plt.plot(x,model.predict(x),color='red',label='Predicted Line')
plt.legend()
plt.title('Linear Regresion')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()
st.pyplot(plt)