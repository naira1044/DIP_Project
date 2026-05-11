import streamlit as st       
import cv2                    
import numpy as np         


from point_color_operations import apply_point_color
from histogram import apply_histogram
from filters_restoration import apply_filters_restoration
from segmentation_detection import apply_seg_edge
from morphology import apply_morphology


st.set_page_config(page_title="DIP Project ", layout="wide")


st.title("Digital Image Processing Tool ")


tasks = {
    "1. Point & Color Operations": ["Addition", "Subtraction", "Division", "Complement", "Change Red Lighting", "Swap R to G", "Eliminate Red"],
    "2. Image Histogram": ["Histogram Stretching (Gray)", "Histogram Equalization (Gray)"],
    "3. Filters & Noise": ["Linear: Average Filter", "Linear: Laplacian Filter", "Non-linear: Maximum", "Non-linear: Minimum", "Non-linear: Median", "Non-linear: Mode (Most Frequent)", "Salt & Pepper: Average", "Salt & Pepper: Median", "Salt & Pepper: Outlier Method", "Gaussian: Image Averaging", "Gaussian: Average Filter"],
    "4. Segmentation & Edge Detection": ["Basic Global Thresholding", "Automatic Thresholding (Otsu)", "Adaptive Thresholding", "Sobel Detector"],
    "5. Mathematical Morphology": ["Image Dilation", "Image Erosion", "Image Opening", "Boundary: Internal", "Boundary: External", "Boundary: Morphological Gradient"]
}


st.sidebar.header("Control Panel")

category = st.sidebar.selectbox("Select Operation Type", list(tasks.keys()))
task = st.sidebar.selectbox("Select Task", tasks[category])

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
   
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

   
    if st.sidebar.button("Apply & Show Result", type="primary"):

        with st.spinner(f'Applying {task}...Please wait...'):

            
            if category == "1. Point & Color Operations":
                input_display, result = apply_point_color(original_img, task)
            elif category == "2. Image Histogram":
                input_display, result = apply_histogram(original_img, task)
            elif category == "3. Filters & Noise":
                input_display, result = apply_filters_restoration(original_img, task)
            elif category == "4. Segmentation & Edge Detection":
                input_display, result = apply_seg_edge(original_img, task)
            elif category == "5. Mathematical Morphology":
                input_display, result = apply_morphology(original_img, task)

            
            def prep_for_display(image):
                if len(image.shape) == 3:       
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image                    

            st.markdown(f"### Results for: `{task}`")
            col1, col2 = st.columns(2)

            with col1:
                
                st.image(prep_for_display(input_display), caption="Input Image", use_container_width=True)

            with col2:
                
                st.image(prep_for_display(result), caption="Result Image", use_container_width=True)

else:
    
    st.info("Please upload an image from the sidebar to start.")
