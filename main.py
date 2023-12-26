import streamlit as st
from PIL import Image
from keras_preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

option = st.selectbox('Model',('MobileNet', 'DenseNet201', 'NasNetLarge'), key = 'tick')
st.write('Your selected:', st.session_state.tick)
if st.session_state.tick == 'MobileNet' :
    model = load_model('modelmobilenet.h5', compile=False)
if st.session_state.tick == 'DenseNet201' :
    model = load_model('modeldensenet201.h5', compile=False)
if st.session_state.tick == 'NasNetLarge' :
    model = load_model('modelnasnetlarge.h5', compile=False)
# lab = {0:'no_tumor', 1:'pituitary_tumor', 2:'meningioma_tumor', 3:'glioma_tumor'}
lab = {0:'pituitary_tumor', 1:'meningioma_tumor', 2:'glioma_tumor', 3:'no_tumor'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img.convert("RGB")
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def processed_img_nasnet(img_path):
    img=load_img(img_path,target_size=(331,331,3))
    img.convert("RGB")
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    st.markdown('''<h4 style='text-align: center; color: #fff;'>BRAIN TUMOR PREDICTION</h4>''',
                unsafe_allow_html=True)
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>Prediction</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of MRI", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        if st.button("Predict"):
            if option=="NasNetLarge":
                result = processed_img_nasnet(save_image_path)
                st.success("Predict: "+result)
            else:
                result = processed_img(save_image_path)
                st.success("Predict: "+result)
        
run()