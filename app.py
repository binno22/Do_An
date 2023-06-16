import streamlit as st
import cv2
import av
import numpy as np
import time
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from utils import *

class VideoTransformer:
    
    def recv(self, frame):
        frame = frame.to_ndarray(format='bgr24')
        frame = detect_image(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return av.VideoFrame.from_ndarray(frame)


demo_img = 'demo.jpg'

# TẠO GIAO DIỆN
st.title('Face Detection App')

st.markdown(
    """
    <style>
    [data-testid=stSidebar][aria-expanded="true] > div:first-child{
        width: 350px
    }
    [data-testid=stSidebar][aria-expanded="false] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Mask Sidebar')
st.sidebar.subheader("Siuuuuuuuuuuuu")


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))

    # resize image
    resized = cv2.resize(imgae, dim, interpolation = inter)

    return resized

# CHỌN PHẦN
app_mode = st.sidebar.selectbox("Choose the App mode",
    ['About App', 'Run on Image', 'Run on Video']
)

if app_mode == 'About App':
    st.markdown('In this Application we are using *CNN* for detect ...')

    st.markdown(
        """
        <style>
        [data-testid=stSidebar][aria-expanded="true] > div:first-child{
            width: 350px
        }
        [data-testid=stSidebar][aria-expanded="false] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.video('https://youtu.be/Ci34un4CVZU')

    st.markdown('''
        # About me \n
          Hey this is **Anh Duc Nguyen** from **HUST**. \n

          If you are interested in building more Computer Vision apps like this one then visit here.\n

          Something
          - [Facebook](https://facebook.com)
          - [LinkedIN](https://linkedin.com)
    ''')

elif app_mode == 'Run on Image':
    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid=stSidebar][aria-expanded="true] > div:first-child{
            width: 350px
        }
        [data-testid=stSidebar][aria-expanded="false] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # UPLOAD ẢNH
    img_file = st.sidebar.file_uploader("Upload an Image", type = ["jpg", "jpeg", "png"])
    if img_file is not None:
        image = np.array(Image.open(img_file))
    else:
        demo_img = 'demo.jpg'
        image = np.array(Image.open(demo_img))

    # ẢNH BAN ĐẦU
    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    # ẢNH SAU KHI ĐƯỢC XỬ LÝ
    output_image = image.copy()

    output_image = detect_image(output_image)

    st.subheader('Detect Face Mask')
    st.image(output_image, use_column_width=True)

elif app_mode == 'Run on Video':
    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid=stSidebar][aria-expanded="true] > div:first-child{
            width: 350px
        }
        [data-testid=stSidebar][aria-expanded="false] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # st.subheader('Webcam App')
    # start = st.button('Start')
    # stop = st.button('Stop')
    # FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(0)

    # run = False

    # while True:
    #     if start:
    #         run = True
    #     if stop:
    #         run = False
    #     if run:
    #         ret, frame = camera.read()
    #         if ret:
    #             frame = detect_image(frame)
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             FRAME_WINDOW.image(frame)
    #         else:
    #             st.error("Không mở được camera")
    #             break
    #     else:
    #         break

    # camera.release()

    st.subheader('Webcam App')
    webrtc_streamer(key="key", video_transformer_factory=VideoTransformer, rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}]})
    )

    # if ret.video_transformer:
    #     transformed_frames = ret.video_transformer.recv()
    #     if transformed_frames is not None:
    #         st.image(transformed_frames.to_image(), channels="BGR")