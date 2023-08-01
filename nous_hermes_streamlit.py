# Import necessary libraries
import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch

# Load local model and tokenizer
model_dir = "C:\\Users\\wesla\\Llama-2-13B-Nous-Hermes-vicuna-uncensored-mastermod-spych"
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Streamlit web UI
st.title("Model Inference System")
input_text = st.text_input("Enter your input text here")

if st.button("Run Inference"):
    if input_text:
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors='pt')

        # Run model
        outputs = model(**inputs)

        # Process output (This would depend on your specific model)
        # This example simply returns the last hidden state from a transformer model
        final_output = outputs.last_hidden_state

        # You can then do what you want with this output, like showing it in the Streamlit app:
        st.write(final_output)
    else:
        st.write("Please enter input text")
