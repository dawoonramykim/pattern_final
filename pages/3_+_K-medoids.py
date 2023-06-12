import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs

st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 기말고사",
)

st.header("K-medoids")
st.markdown("")
st.markdown("K-medoids는 K-means를 변형한 것으로, 군집의 무게 중심을 구하기 위해서 데이터의 :red[평균 대신 중간점](medoids)을 사용하는 알고리즘이다.")

st.sidebar.markdown("K-medoids의 :red[변수] ")
number = st.sidebar.number_input('How many center do you want?',1,6,3)
st.write('The current center is ', number)
k=st.sidebar.selectbox(
    'How many K do you want?',
    (1,2,3,4,5,6),2)
st.write("The current K is",k)

data,true_labels=make_blobs(n_samples=100,centers=number,cluster_std=0.6,random_state=0)

kmedoids=KMedoids(n_clusters=k,random_state=0).fit(data)

for j in range(k):
    cluster_data=data[kmedoids.labels_==j]
    plt.scatter(cluster_data[:,0],cluster_data[:,1],label=f"Cluster{j+1}")
    
plt.scatter(kmedoids.cluster_centers_[:,0],kmedoids.cluster_centers_[:,1],marker="x",color="k",s=100,label="Medoids")
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("K_medoids Clustering")
#plt.show()
st.pyplot(plt)
