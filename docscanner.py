import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageEnhance
from PIL import Image,ImageEnhance
import requests
import io
import json
from utils import image_grid, width, height, draw_text, biggest_contour, draw_rectangle, reorder

img = cv2.imread("scanned_doc.jpg")
#PATH = "2.jpg"
def ocrDetect(img):
    img = cv2.imread("scanned_doc.jpg")
    height, width, _ = img.shape
    roi = img
    url_api = "https://api.ocr.space/parse/image"
    _, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
    file_bytes = io.BytesIO(compressedimage)
    result = requests.post(url_api,
              files = {"screenshot.jpg": file_bytes},
              data = {"apikey": "helloworld",
                      "language": "eng"})
    result = result.content.decode()
    result = json.loads(result)

    parsed_results = result.get("ParsedResults")[0]
    text_detected = parsed_results.get("ParsedText")
    st.write(text_detected)
    #cv2.imshow("roi", roi)
    #cv2.imshow("Img", img)

#image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
#our_image = Image.open(image_file)
PATH="13.jpg"

def doc_scan_pipeline(input= PATH,  output="scanned_doc.jpg"):
    img = cv2.imread(input)

    # 0. Convert given image from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    img = cv2.resize(img, (width, height))

    # 1. Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Add Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    # 3. Add Canny edge detection
    img_threshold = cv2.Canny(img_blur, 100, 200, L2gradient=True)

    # 3.1 Apply dilation
    kernel = np.ones((3, 3))
    img_threshold = cv2.dilate(img_threshold, kernel, iterations=2)

    # 4. Find all the contours
    img_contours = img.copy()
    img_big_contour = img.copy()
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5)

    # 5. Find the biggest contour
    biggest, maxArea = biggest_contour(contours)
    biggest = reorder(biggest)
    cv2.drawContours(image=img_big_contour, contours=biggest, contourIdx=-1, color=(0, 255, 0), thickness=10)

    # 5.1 Draw a rectangle, i.e., 4 lines connecting the 4 dots corresponding to the largest contour
    img_big_contour = draw_rectangle(img_big_contour, biggest, thickness=2)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # 6. Image Warp
    # 6.1 Calculate a 3x3 perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 6.2 Apply the perspective matrix to the image
    img_warp_coloured = cv2.warpPerspective(img, matrix, (width, height))

    # 7. Adaptive thresholding
    img_warp_gray = cv2.cvtColor(img_warp_coloured, cv2.COLOR_BGR2GRAY)
    img_adaptive_th = cv2.adaptiveThreshold(img_warp_gray, 255, 1, cv2.THRESH_BINARY, 5, 2)

    # 7.1 Apply median blurring to remove tiny speckles of noise
    img_adaptive_th = cv2.medianBlur(img_adaptive_th, 3)

    # Save the document to disk
    cv2.imwrite(output, img_adaptive_th)

    # Add labels to each image
    img = draw_text(img, "Original")
    img_gray = draw_text(img_gray, "Grayscale")
    img_blur = draw_text(img_blur, "Gaussian Blur", pos=(int(width / 4), 50))
    img_threshold = draw_text(img_threshold, "Canny Edge", pos=(int(width / 4), 50))
    img_contours = draw_text(img_contours, "Contours")
    img_big_contour = draw_text(img_big_contour, "Largest Contour", pos=(int(width / 7), 50))
    img_warp_coloured = draw_text(img_warp_coloured, "Warp", pos=(int(width / 3), 50))
    img_adaptive_th = draw_text(img_adaptive_th, "Adaptive Thresholding", pos=(int(width / 7), 50),
                                font_scale=2, font_thickness=6)

    blank_img = np.zeros((height, width, 3), dtype=np.uint8)
    image_list = [img, img_gray, img_blur, img_threshold, img_contours, img_big_contour, img_warp_coloured, img_adaptive_th]

    # Combine the images into a grid
    # image_grid returns PIL image, np.asarray() can be used to convert it back to cv2 compatible format
    grid = np.asarray(image_grid(image_list, width, height))


def main():
    st.sidebar.markdown(
        """<img style=' align:center;  display: block;margin-left: auto;margin-right: auto;width: 70%; padding-bottom: 20%' src="https://media.glassdoor.com/sqll/2373691/valhalla-online-services-squarelogo-1544426755364.png">""",
        unsafe_allow_html=True)

    st.title("Virtual NotePad")
    st.text("Buils with OpenCv and Streamlit")
    activities=["Detection","About"]
    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == 'Detection':
        st.subheader("Text Detection")

        image_file = "13.jpg"               #st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
        # st.write(type(our_image))
            st.image(our_image)
        task = ['text','ocr']

        feature_choice = st.sidebar.selectbox("Find Features",task)
        enhance_type = st.sidebar.radio("Enhance Type",
                                        ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # st.write(new_img)
            st.image(gray)

        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        if enhance_type == 'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            img = cv2.cvtColor(new_img, 1)
            blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            st.image(blur_img)
        if  st.button('Process'):
            if feature_choice == 'text':
                doc_scan_pipeline(input='13.jpg', output="scanned_doc.jpg")
                result_image="scanned_doc.jpg"
                st.image(result_image)

            elif feature_choice == 'ocr':
                st.subheader("Extracted Text")
                img = cv2.imread("scanned_doc.jpg")
                height, width, _ = img.shape
                roi = img
                url_api = "https://api.ocr.space/parse/image"
                _, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
                file_bytes = io.BytesIO(compressedimage)
                result = requests.post(url_api,
                                       files={"screenshot.jpg": file_bytes},
                                       data={"apikey": "helloworld",
                                             "language": "eng"})
                result = result.content.decode()
                result = json.loads(result)

                parsed_results = result.get("ParsedResults")[0]
                text_detected = parsed_results.get("ParsedText")
                st.write(text_detected)
                print(text_detected)


    elif choice =='About':
        st.title('ABOUT')

        st.title("TEAM VALHALLA")





if __name__ == "__main__":
    main()
    #doc_scan_pipeline(input="2.jpg", output="./img/scanned_doc.jpg")
    # doc_scan_pipeline(input="./img/tnk_art.jpg", output="./img/tnk_scan.jpg")