import streamlit as st
from PIL import Image
import os
import numpy as np
from numpy.linalg import norm
import pickle 
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors


model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array= image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result=result/norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbours=NearestNeighbors(n_neighbors=6,algorithm="brute",metric="euclidean")
    neighbours.fit(feature_list)
    distances,indices=neighbours.kneighbors([features])
    return indices

st.title('Clothes Recommender System')

directory_path = 'data\manyavar_data'

list_gen=[]
men_type=[]
women_type=[]
for genders in os.listdir(directory_path):
    list_gen.append(genders)
    
for mc in os.listdir(directory_path+'\men'):
    men_type.append(mc)

for wc in os.listdir(directory_path+'\women'):
    women_type.append(wc)

selected_gender = st.selectbox(
    "Select Gender",
    ["Select Gender"] + list_gen,
    
    format_func=lambda x: "Select Gender" if x == "Select Gender" else x
)
if selected_gender ==list_gen[0]:
    selected_clothes = st.selectbox(
    "Select Clothes",
    ["Select Clothes"] + men_type,
    format_func=lambda x: "Select Clothes" if x == "Select Clothes" else x
    
)
elif selected_gender ==list_gen[1]:
    selected_clothes = st.selectbox(
    "Select Clothes",
    ["Select Clothes"] + women_type,
    format_func=lambda x: "Select Clothes" if x == "Select Clothes" else x
)
try:
    if selected_clothes=="Indo-western":
            feature_list=np.array(pickle.load(open('embedings/men/indo-west.pkl','rb')))
            filenames=pickle.load(open('image_names/men/indo-west.pkl','rb'))
    elif selected_clothes=="jacket":
            feature_list=np.array(pickle.load(open('embedings/men/jacket.pkl','rb')))
            filenames=pickle.load(open('image_names/men/jacket.pkl','rb'))
    elif selected_clothes=="kurta dhoti":
            feature_list=np.array(pickle.load(open('embedings/men/kurta-dhoti.pkl','rb')))
            filenames=pickle.load(open('image_names/men/kurta-dhoti.pkl','rb'))
    elif selected_clothes=="kurta jacket set":
            feature_list=np.array(pickle.load(open('embedings/men/kurta-jacket-set.pkl','rb')))
            filenames=pickle.load(open('image_names/men/kurta-jacket-set.pkl','rb'))
    elif selected_clothes=="kurta":
            feature_list=np.array(pickle.load(open('embedings/men/kurta.pkl','rb')))
            filenames=pickle.load(open('image_names/men/kurta.pkl','rb'))
    elif selected_clothes=="suits":
            feature_list=np.array(pickle.load(open('embedings/men/suits.pkl','rb')))
            filenames=pickle.load(open('image_names/men/suits.pkl','rb'))
    elif selected_clothes=="Gown":
            feature_list=np.array(pickle.load(open('embedings/women/gown.pkl','rb')))
            filenames=pickle.load(open('image_names/women/gown.pkl','rb'))
    elif selected_clothes=="saree":
            feature_list=np.array(pickle.load(open('embedings/women/saree.pkl','rb')))
            filenames=pickle.load(open('image_names/women/saree.pkl','rb'))
    elif selected_clothes=="Stitched suit":
            feature_list=np.array(pickle.load(open('embedings/women/Stitched-suit.pkl','rb')))
            filenames=pickle.load(open('image_names/women/Stitched-suit.pkl','rb'))
except:
    pass


def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        image = Image.open(uploaded_file)
        image.save(file_path, 'JPEG')
        return 1
    except Exception as e:
        print(f"Error saving file: {e}")
        return 0

try:
    if selected_clothes !="Select Clothes":
        uploaded_file = st.file_uploader("Choose an image")
        if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
                display_img=Image.open(uploaded_file)
                st.image(display_img,width=200)
                features=extract_features(os.path.join("uploads",uploaded_file.name),model)
                # st.text(features)
                indices=recommend(features,feature_list)
                col1,col2,col3,col4,col5=st.columns(5)
                with col1:
                    st.image(filenames[indices[0][0]])
                with col2:
                    st.image(filenames[indices[0][1]])
                with col3:
                    st.image(filenames[indices[0][2]])
                with col4:
                    st.image(filenames[indices[0][3]])
                with col5:
                    st.image(filenames[indices[0][4]])
            else:
                st.header("Some error with uploading")
except:
    pass

