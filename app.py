import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, BertTokenizerFast

@st.cache(allow_output_mutation=True)
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base")
    tokenizer = BertTokenizerFast.from_pretrained("microsoft/trocr-base")
    return model, tokenizer

def app():
    model, tokenizer = load_model()

    st.title("TrOCR Demo")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        inputs = tokenizer(images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        st.write("OCR Output:")
        st.write(prediction)

if __name__ == "__main__":
    app()
