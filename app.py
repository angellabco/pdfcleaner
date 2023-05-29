import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import cv2
import os

st.title('PDF OCR App')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        images = convert_from_path('temp.pdf')
        for i in range(len(images)):
            # Save pages as images in the pdf
            images[i].save('page'+ str(i) +'.jpg', 'JPEG')
    except:
        st.write("Error in converting pdf to image")
    st.write('Converted PDF to Images')

    text = ''
    for i in range(len(images)):
        # Open image using OpenCV
        img = cv2.imread('page'+ str(i) +'.jpg')

        # Preprocess the image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin

        # Save the preprocessed image
        cv2.imwrite('preprocessed_page'+ str(i) +'.jpg', img_bin)

        # Perform OCR on the preprocessed image
        text += pytesseract.image_to_string(Image.open('preprocessed_page'+ str(i) +'.jpg'))

    st.write('Performed OCR on images')
    st.text_area("Text", text)

    # Clean up temporary files
    os.remove("temp.pdf")
    for i in range(len(images)):
        os.remove('page'+ str(i) +'.jpg')
        os.remove('preprocessed_page'+ str(i) +'.jpg')
