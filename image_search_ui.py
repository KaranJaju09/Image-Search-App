import streamlit as st
from pymilvus import MilvusClient
import clip
from PIL import Image
import torch
import os
import glob

# --- Database Initialization ---

@st.cache_resource
def initialize_once():
    """
    Initializes the Milvus database by running the embedding script.
    This function is cached to ensure it only runs once per session.
    """
    from embed_images_to_milvus import initialize_database
    initialize_database()

# Ensure the database is initialized before using the app
initialize_once()

# --- Configuration ---

MILVUS_URI = "./milvus.db"
COLLECTION_NAME = "image_embeddings"
MODEL_NAME = "ViT-B/32"
TEST_FOLDER = "images_folder/test"

# --- Milvus and CLIP Model Initialization ---

@st.cache_resource
def get_milvus_client():
    """
    Creates and returns a Milvus client.
    The client is cached to avoid reconnecting on every interaction.
    """
    return MilvusClient(uri=MILVUS_URI)

milvus_client = get_milvus_client()

@st.cache_resource
def get_clip_model():
    """
    Loads and returns the CLIP model and its preprocessor.
    The model is cached for performance.
    """
    device = "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    return model, preprocess, device

model, preprocess, device = get_clip_model()

# --- Image Processing ---

def encode_image(image):
    """
    Encodes a single image using the CLIP model.

    Args:
        image (PIL.Image): The image to encode.

    Returns:
        list: The normalized image embedding as a list of floats.
    """
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().squeeze().tolist()

def load_test_images(folder):
    """
    Recursively loads all images from a specified folder.

    Args:
        folder (str): The path to the folder containing the images.

    Returns:
        list: A list of file paths for the loaded images.
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    return [f for f in glob.glob(os.path.join(folder, '**', '*'), recursive=True) if os.path.splitext(f)[-1].lower() in image_extensions]

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Image Search App")

st.title("Image Search with CLIP and Milvus")

# Check if the Milvus collection exists
has_collection = milvus_client.has_collection(collection_name=COLLECTION_NAME)
if not has_collection:
    st.error(f"Milvus collection '{COLLECTION_NAME}' not found. Please run `embed_images_to_milvus.py` first to populate the database.")
    st.stop()

# --- Search Mode Selection ---

search_mode = st.radio("Choose search mode:", ["Search using gallery", "Upload an image"])

num_results = st.slider("Number of results to display:", min_value=1, max_value=10, value=5)

selected_image = None
selected_image_path = None

# --- Gallery Search ---

if search_mode == "Search using gallery":
    st.subheader("Select an image from the gallery")
    test_images = load_test_images(TEST_FOLDER)

    if not test_images:
        st.warning("No images found in the test folder.")
    else:
        # Display images in a grid
        cols = st.columns(5)
        for idx, image_path in enumerate(test_images):
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))  # Resize for consistent display
            with cols[idx % 5]:
                if st.button(f"Select Image {idx+1}", key=f"img_{idx}"):
                    selected_image_path = image_path
                st.image(img, caption=os.path.basename(image_path), use_container_width=True)

# --- Upload Search ---

elif search_mode == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image for search", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file).convert("RGB")

# --- Perform Search and Display Results ---

if selected_image or selected_image_path:
    st.subheader("Search Results")

    # Determine the query image
    if selected_image_path:
        image = Image.open(selected_image_path).convert("RGB")
    else:
        image = selected_image

    with st.spinner("Encoding image and searching Milvus..."):
        try:
            # Encode the query image and search in Milvus
            query_embedding = encode_image(image)
            search_results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_embedding],
                limit=num_results,
                output_fields=["image_path"],
            )

            # Display the search results
            if search_results and search_results[0]:
                st.subheader("Relevant Images:")
                cols = st.columns(5)
                col_idx = 0
                for hit in search_results[0]:
                    image_path = hit["entity"]["image_path"]
                    distance = hit["distance"]
                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path)
                            with cols[col_idx % 5]:
                                st.image(img, caption=f"Score: {distance:.4f}", use_container_width=True)
                            col_idx += 1
                        except Exception as e:
                            st.warning(f"Could not load image {image_path}: {e}")
                    else:
                        st.warning(f"Image file not found: {image_path}")
            else:
                st.info("No results found.")
        except Exception as e:
            st.error(f"An error occurred during search: {e}")

# --- Sidebar ---

st.sidebar.header("About")
st.sidebar.info("This is an image-to-image search application using CLIP and Milvus.")
st.sidebar.info("Select or upload an image to find similar results from the indexed dataset.")
