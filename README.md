# Image Search with CLIP and Milvus

This project implements an image search application using OpenAI's CLIP model for generating image embeddings and Milvus as a vector database for efficient similarity search. The user interface is built with Streamlit.

## Live Application

If you want to try out the app, you can visit this link to use it - https://image-search-app-kj.streamlit.app/ 

**Note**: The app may be in sleep mode due to inactivity, user should please wait the app to spawn up, it may take upto half an hour.

## Features

-   **Image Indexing:** Recursively scans a directory of images, generates CLIP embeddings for each image, and stores them in a Milvus collection.
-   **Image Search:** Allows users to either upload an image or select one from a gallery to find similar images from the indexed collection.
-   **Streamlit UI:** Provides a user-friendly web interface for interacting with the application.

## Getting Started

### Prerequisites

-   Python 3.7+
-   Pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/KaranJaju09/Image-Search-App.git
    cd image-to-image-search
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the training data:**

    Use the following commands to download the sample dataset of images:

    ```bash
    wget https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip
    unzip reverse_image_search.zip
    ```

    Download the training data and extract it into the `images_folder/train` directory.

### Usage

**Run the Streamlit application:**

```bash
streamlit run image_search_ui.py
```

The application will automatically index the images in the `images_folder/train` directory and start the Streamlit server. You can then open the application in your web browser.

## Project Structure

```
.
├── embed_images_to_milvus.py   # Script to index images and store embeddings in Milvus
├── image_search_ui.py          # Streamlit application for the image search UI
├── requirements.txt            # Python dependencies
├── images_folder/              # Contains the dataset of images
│   ├── train/                  # Images for indexing
│   └── test/                   # Images for the gallery in the UI
└── milvus.db                   # Local Milvus database file
```

## Dependencies

The project's dependencies are listed in the `requirements.txt` file:

-   `streamlit`
-   `pymilvus`
-   `torch`
-   `openai-clip`
-   `Pillow`
-   `torchvision`