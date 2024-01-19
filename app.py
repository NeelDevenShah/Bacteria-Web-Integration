import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import pickle

labels = {0: 'Acinetobacter.baumanii', 1: 'Actinomyces.israeli', 2: 'Bacteroides.fragilis', 3: 'Bifidobacterium.spp', 4: 'Candida.albicans', 5: 'Clostridium.perfringens', 6: 'Enterococcus.faecalis', 7: 'Enterococcus.faecium', 8: 'Escherichia.coli', 9: 'Fusobacterium', 10: 'Lactobacillus.casei', 11: 'Lactobacillus.crispatus', 12: 'Lactobacillus.delbrueckii', 13: 'Lactobacillus.gasseri', 14: 'Lactobacillus.jehnsenii', 15: 'Lactobacillus.johnsonii', 16: 'Lactobacillus.paracasei',
          17: 'Lactobacillus.plantarum', 18: 'Lactobacillus.reuteri', 19: 'Lactobacillus.rhamnosus', 20: 'Lactobacillus.salivarius', 21: 'Listeria.monocytogenes', 22: 'Micrococcus.spp', 23: 'Neisseria.gonorrhoeae', 24: 'Porfyromonas.gingivalis', 25: 'Propionibacterium.acnes', 26: 'Proteus', 27: 'Pseudomonas.aeruginosa', 28: 'Staphylococcus.aureus', 29: 'Staphylococcus.epidermidis', 30: 'Staphylococcus.saprophiticus', 31: 'Streptococcus.agalactiae', 32: 'Veionella'}

# Load pre-trained deep learning model (Replace 'path_to_your_model' with the actual path)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print('Model loaded')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title("Bacteria Image Classifier")
st.divider()
# Upload image through Streamlit
uploaded_file = st.file_uploader(
    "Choose a bacteria image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Display the uploaded image

    image = Image.open(uploaded_file)
    # expand image if image size is less than 224 and converting it to RGB
    if image.size[0] < 224 or image.size[1] < 224:
        image = image.resize((224, 224))
    image = image.convert('RGB')
    st.image(image, caption="Uploaded Image", width=200)

if st.button('Predict'):

    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    model.eval()
    # Inference
    out = model(batch_t)

    # Print predictions
    max_score, idx = torch.max(out, 1)
    st.divider()
    # Display the prediction result
    st.write("Prediction:")
    st.markdown(
        f"Class: <code>{labels[idx.item()]}</code>, Score: <code>{max_score.item()}</code>", unsafe_allow_html=True)

# About this project
st.divider()
st.header("About This Project")
st.markdown(
    """
    This Streamlit app is designed for predicting bacteria types based on uploaded images. It utilizes a pre-trained deep learning model to make predictions.

    - **Model:** EfficientNet-B0
    - **Image:** images are resized to 224x224 pixels.
    """
)

# Contributors section
st.divider()
st.header("This Project is Made By: ")

# Define contributors and their GitHub profile links
contributors = {
    "Pankil Soni": "https://github.com/pankil-soni/",
    "Neel Shah": "https://github.com/NeelDevenShah",
    "Sneh Shah": "https://github.com/Sneh-T-Shah/",
}

# Display contributors with GitHub icons and links in one line


colli = st.columns(3)

with colli[0]:
    st.markdown(
        f'''<a href="{contributors['Neel Shah']}"><img src="https://avatars.githubusercontent.com/u/106433515?v=4" width="100px"><br><a href = {contributors['Neel Shah']} style="margin: 12px;">Neel Shah</a>''', unsafe_allow_html=True)

with colli[1]:
    st.markdown(f'''<a href="{contributors['Pankil Soni']}"><img src="https://avatars.githubusercontent.com/u/116267467?v=4" width="100px"><br><a href = {contributors['Pankil Soni']} style="margin: 12px;">Pankil Soni</a>''', unsafe_allow_html=True)

with colli[2]:
    st.markdown(f'''<a href="{contributors['Sneh Shah']}"><img src="https://avatars.githubusercontent.com/u/120238003?s=48&v=4" width="100px"><br><a href = {contributors['Sneh Shah']} style="margin: 12px;">Sneh Shah</a>''', unsafe_allow_html=True)

st.markdown("Department Of Artificial Intelligence And Machine Learning, Chandubhai S. Patel Institute Of Technology, Charusat University")