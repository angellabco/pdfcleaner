import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the TrOCR model
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base")
    return model, processor

def app():
    model, processor = load_model()

    st.title('TrOCR Application')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')

        # Convert image to the format required by TrOCR
        # Here, a simple conversion is done. More preprocessing might be required depending on the image
        image = processor(document=[image], return_tensors="pt", padding=True)

        # Perform OCR with the TrOCR model
        predictions = model.generate(input_ids=image.input_ids, attention_mask=image.attention_mask, decoder_input_ids=image.decoder_input_ids, decoder_attention_mask=image.decoder_attention_mask)

        # Decode the generated tensor into human-readable text
        predicted_text = processor.batch_decode(predictions, skip_special_tokens=True)

        # Display the OCR results
        st.write('Extracted Text:')
        st.write(predicted_text[0])

if __name__ == '__main__':
    app()
