import streamlit as st
from PIL import Image
from transformers import TrOCRForCausalLM, TrOCREncoderModel
from transformers import VisionEncoderDecoderModel

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the TrOCR model
    encoder = TrOCREncoderModel.from_pretrained("microsoft/trocr-base")
    decoder = TrOCRForCausalLM.from_pretrained("microsoft/trocr-base")
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    return model

def app():
    model = load_model()

    st.title('TrOCR Application')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')

        # Convert image to the format required by TrOCR
        # Here, a simple conversion is done. More preprocessing might be required depending on the image
        image = image.convert('RGB')

        # Perform OCR with the TrOCR model
        inputs = model.prepare_inputs_for_generation(image)
        predictions = model.generate(**inputs)

        # Display the OCR results
        st.write('Extracted Text:')
        st.write(predictions)

if __name__ == '__main__':
    app()
