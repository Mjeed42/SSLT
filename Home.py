import streamlit as st
from streamlit_option_menu import option_menu
import time

st.set_page_config(
    page_title="ASLT",
    page_icon="🤖",
    layout="wide"
    )

# Custom CSS to make it look even better
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #FF4B4B;
    }
    .SideBar {
            font-size: 58px;
            font-weight: bold;
            text-align: center;
            background: -webkit-linear-gradient(45deg, #004474, #ff5733, #28a745);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.markdown(
    "<h1 class = 'SideBar' style='color:orange; font-size:70px;'>ASLT</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown("----")

search_term = st.sidebar.text_input("Search", "")
if search_term:
    st.write(f"Searching for: {search_term}")

# Fancy horizontal menu
selected = option_menu(
    menu_title=None,
    options=["ℹ️ About", "👥 Team Members"],
    # icons=["house", "code-slash", "envelope"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        # "icon": {"color": "black", "font-size": "20px"},
        "nav-link": {
            "font-size": "20px",
            "text-align": "center",
            "color": "black",
            "margin": "0px",
            "--hover-color": "#b8b8b884",
            "transition": "all 0.3s ease"
        },
        "nav-link-selected": {"background-color": "#FF4B4B"},
    }
)

# Page content
#--------------------------------Home----------------------------------
if selected == "ℹ️ About":
    col1, col2 = st.columns([1,4])
    with col1:
        st.image("Pictures/LewagonxSDA.png")
    with col2:
        st.markdown("---")

    st.title("🤖 Real-Time Arabic Sign Language Translator")
    st.markdown("---")
    st.markdown("""<h2 style='font-size:24px;'>This project builds a real-time Arabic Sign Language translator that captures hand gestures using a camera and instantly translates them into text or speech.</h2>
                 """, unsafe_allow_html=True)





    st.markdown("---")
    st.title("""
    🔎 Project Overview:
    """)
    st.markdown("""<h2 style='font-size:24px;'>The Arabic Sign Language Translation Project is an AI-powered initiative designed to facilitate real-time communication between the deaf community and the broader society. The system captures hand gestures using a standard camera, interprets them using deep learning models, and instantly converts them into spoken Arabic.</h2>
                 """, unsafe_allow_html=True)
    st.markdown("""<h2 style='font-size:24px;'>By combining computer vision, convolutional and recurrent neural networks, and speech synthesis, the application provides a seamless and intuitive user experience. Our goal is to bridge the communication gap and foster inclusivity by making everyday interactions more accessible for individuals who use Arabic Sign Language.</h2>
                 """, unsafe_allow_html=True)
    st.markdown("""<h2 style='font-size:24px;'>This project represents a meaningful step toward leveraging technology to promote social equity, break down communication barriers, and support the empowerment of the deaf and hard-of-hearing community across the Arabic-speaking world.</h2>
                 """, unsafe_allow_html=True)
    
  
    st.markdown("---")
    st.title("""
    📊 Statistics :
    """)
    st.markdown("""<h2 style='font-size:24px;'>According to the 2022 Population and Housing Census, the Kingdom of Saudi Arabia has a total population of 32,175,224 people, with 1,349,585 individuals with disabilities (7.1%).</h2>
                 """, unsafe_allow_html=True)
    st.image(
        "Pictures/Pic1.png",
        caption="Statistic from APD shows percentage of disabled people in Saudi Arabia",
        use_container_width=True,
        width=300
    )
    st.markdown("---")
    
    st.image(
        "Pictures/Pic2.png",
        caption="picture showing each disability by numbers in Saudi Arabia",
        use_container_width=True,
        width=300
    )
    st.markdown("---")
    st.markdown("<h1 style='font-size:36px;'>Our Focus</h1>", unsafe_allow_html=True)
    st.image(
        "Pictures/Pic3.png",
        use_container_width=True,
        width=300
    )

    st.markdown("---")
    st.markdown("""
    ### ⚙️ How it Works:
    1. **Capture**: The system uses a camera to capture hand gestures in real-time.
    2. **Detect**: OpenCV detects and localizes hands in each video frame.
    3. **Recognize**: A Convolutional Neural Network (CNN) extracts visual features from hand shapes and gestures.
    4. **Translate**: A Recurrent Neural Network (RNN) understands sequences of gestures over time for dynamic sign translation.
    5. **Output**: The recognized text is converted into clear and natural spoken audio using Text-To-Speech (TTS) technology.
    6. **Deploy**: The system is containerized using Docker and hosted on Google Cloud Platform (GCP) for reliable cloud performance.
    7. **Interface**: An interactive web interface is built using Streamlit for real-time use.

    """)

    st.markdown("---")
    # Features Section
    st.markdown("""
    ### 🔥 Features
    - **User-Friendly Interface**: Easy-to-use web interface for seamless interaction.

    - **Scalable**: Deployed on GCP for reliable performance and scalability.
    - **Cross-Platform**: Accessible from any device with a web browser.

    - **Educational Resource**: Aims to raise awareness and understanding of Arabic sign language.

    - **Future Enhancements**: Plans to integrate additional features like gesture recognition for different sign languages and dialects.
    - **Real-Time Feedback**: Provides users with instant feedback on their gestures to improve learning.

    - **Gesture Library**: A comprehensive library of gestures and their meanings for reference.
    - **Offline Mode**: Future plans to enable offline functionality for areas with limited internet access.

    - **Educational Tool**: Aims to be a valuable resource for learning Arabic sign language.
    - **User-Centric Design**: Focused on creating a user-friendly experience for both deaf individuals and the wider community.

    - **Continuous Improvement**: Committed to ongoing development and refinement of the system based on user feedback and technological advancements.


    """)
    st.markdown("---")
    st.markdown("""
    ### 🧠 Technologies:

    - 📷 **OpenCV**: Detects and localizes hands in each video frame with high speed and precision.
    - 👐 **MediaPipe **: Detects hand landmarks with high precision for gesture recognition.
    - 🔄 **LSTM Network**: Understands sequences of gestures over time for dynamic sign translation.
    - 🤖 **TensorFlow/Keras**: Powers the deep learning model for accurate sign classification.
    - 💬 **Text-To-Speech (TTS)**: Converts recognized sign language into natural spoken Arabic using streamlit_TTS + gTTS.
    - 📊 **Pandas**: Handles data manipulation and preprocessing for the machine learning model.
    - 📈 **Scikit-learn**: Provides tools for model evaluation and performance metrics.
    - 📦 **Streamlit**: Creates an interactive web interface for real-time use.


    """)
    st.markdown("---")
    st.markdown("""

    ### 🚦 Limitations:
     - Lack of Dataset: One of the primary obstacles was the lack of a publicly available dataset for Saudi Sign Language.
     - Large and Difficult Dataset: The dataset we used was large and difficult to manage, which added complexity to the project.
     - Impact on Model Accuracy: The quality of the dataset negatively impacted the model’s accuracy, limiting its performance.
     - Time and Resource Limitations: We faced challenges in terms of time and resources, which restricted our ability to refine and improve the project further.
     - Educational Experience: Despite these challenges, the experience has been highly educational.
     - Future Enhancements: We hope to build on this foundation and enhance the project in the future.
    """)

    st.markdown("---")
    st.markdown("""
    ### 🌟 Goal:
    > Bridge communication between deaf individuals and the wider community using cutting-edge AI.

    """, unsafe_allow_html=True)


    # search_term = st.sidebar.text_input("Search", "")
    # if search_term:
    #     st.write(f"Searching for: {search_term}")

    # st.snow()
    # st.toast("Welcome to my fancy multi-page app! 🎉", icon="🎈")




#--------------------------------Projects----------------------------------
# elif selected == "💻 Projects":
#     st.title("💻 My Awesome Projects")
#     with st.spinner('Loading projects...'):
#         time.sleep(0)
#     st.success('Projects loaded! 🚀')
#     st.write("Here are some cool things I'm working on:")
#--------------------------------Contact----------------------------------
elif selected == "👥 Team Members":
    col1, col2 = st.columns([1,4])
    with col1:
        st.image("Pictures/LewagonxSDA.png")
    with col2:
        st.markdown("---")

    st.title("The People Behind the Project")
    st.markdown("---")
    st.title("Our Team:")
    st.header("Fayadh ")
    col1, col2 ,col3 , col4 = st.columns(4)
    #----------------------------------------Fayadh-----------------------------
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://www.linkedin.com/in/fayadh3"  target="_blank" style="margin-right: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="mailto:fayadh3@gmail.com" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://github.com/fayadh3" target="_blank" style="margin-right: 10px;">
                    <img src="https://img.icons8.com/ios11/512/FFFFFF/github.png" alt="GitHub" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="https://lewagon-alumni.slack.com/team/U08DZL4LA2D" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg" alt="Slack" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.header("Abdulmajeed ")
    col1, col2 ,col3 , col4 = st.columns(4)
    #----------------------------------------Abdulmajeed-----------------------------
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://www.linkedin.com/in/abdulmajeed-alrashidi-0039492bb/" target="_blank" style="margin-right: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="mailto:alrashabdulmajeed@gmail.com" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://github.com/Mjeed42" target="_blank" style="margin-right: 10px;">
                    <img src="https://img.icons8.com/ios11/512/FFFFFF/github.png" alt="GitHub" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="https://slack.com/your-workspace" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg" alt="Slack" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.header("Bander ")
    col1, col2 ,col3 , col4 = st.columns(4)
    #----------------------------------------Bander-----------------------------
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://www.linkedin.com/in/bandar-al-qahtani-415a00296/overlay/about-this-profile/" target="_blank" style="margin-right: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="mailto:bandr.khald.q@gmail.com" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://github.com/Bandr505q" target="_blank" style="margin-right: 10px;">
                    <img src="https://img.icons8.com/ios11/512/FFFFFF/github.png" alt="GitHub" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="https://lewagon-alumni.slack.com/team/U08BSJAP2R5" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg" alt="Slack" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.header("Saud")
    col1, col2 ,col3 , col4 = st.columns(4)
    #----------------------------------------Saud-------------------------------
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="http://www.linkedin.com/in/saud-sahli-4a53142a1" target="_blank" style="margin-right: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="mailto:SaudSahli2002@gmail.com" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://github.com/SaudSahli2002" target="_blank" style="margin-right: 10px;">
                    <img src="https://img.icons8.com/ios11/512/FFFFFF/github.png" alt="GitHub" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="https://lewagon-alumni.slack.com/team/U08DEV84QUU" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg" alt="Slack" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.header("Saad")
    col1, col2 ,col3 , col4 = st.columns(4)
    #----------------------------------------Saad-------------------------------
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://www.linkedin.com/in/saad-s-alshahrani/" target="_blank" style="margin-right: 10px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="mailto:saadalshahrani111@outlook.com" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
                <a href="https://github.com/SaadAlshahrani" target="_blank" style="margin-right: 10px;">
                    <img src="https://img.icons8.com/ios11/512/FFFFFF/github.png" alt="GitHub" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="display: flex; align-items: left; justify-content: left;">
                <a href="https://lewagon-alumni.slack.com/team/U08DJ8197JR" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg" alt="Slack" style="width:50px;height:50px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
