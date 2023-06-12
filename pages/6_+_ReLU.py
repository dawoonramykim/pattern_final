import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 기말고사",
)

st.header("ReLU(렉티파이어)")
st.markdown("")
st.markdown("기울기 소실 문제를 해결하기 위해 등장한 함수이다. ReLU 함수는 임력값이 0보다 작을 경우에는 0을 반환하고, 0보다 클 경우에는 입력값을 그대로 반환한다. 즉 x가 0보다 클 때, 미분 값은 항상 1ㅇ다. 그래서 층이 아무리 깊어져도 손실 없이 정보를 전달할 수 있다. 미분 값이 0과 1이기 때문에 연산이 빠르다. 하지만 0 이하의 값이 그대로 보존되지 않고 그대로 버려진다는 것이 단점이다.")

def relu(x):
    return np.where(x>0,x,0)

min = st.sidebar.number_input('설정할 최솟값은?(-10<x<0)',-5,0,-5)
max = st.sidebar.number_input('설정할 최댓값은?(0<x<10)',0,5,5)
st.write('선택 범위 : ',min,max)

xx=np.linspace(min,max,50)
plt.plot(xx,relu(xx))
plt.plot(0,0,marker='o',ms=10)
plt.title("ReLU")
plt.xlabel("$x$")
plt.ylabel("$ReLU(x)$")
#plt.show()
st.pyplot(plt)