# Design Document: Image-to-Image Search

## 1. Project Overview

This document outlines the design and architecture of the Image-to-Image Search application. The project's goal is to provide a simple and efficient way to find similar images within a dataset based on a query image. It leverages the power of OpenAI's CLIP model for creating semantic vector embeddings of images and uses Milvus as a high-performance vector database for storing and searching these embeddings.

## 2. Project Structure

```
.
├── embed_images_to_milvus.py   # Handles the image embedding and database population
├── image_search_ui.py          # The main Streamlit application for the UI
├── requirements.txt            # Project dependencies
├── images_folder/              # Contains the image dataset
│   ├── train/                  # Images to be indexed
│   └── test/                   # Images for the search gallery
└── milvus.db                   # Local Milvus database file
```

## 3. Architecture

The application's architecture can be broken down into two main phases:

1.  **Indexing Phase:** The `embed_images_to_milvus.py` script is responsible for this phase. It processes all images in the `images_folder/train` directory, generates a vector embedding for each using the CLIP model, and then inserts these embeddings into a Milvus collection. This phase is automatically triggered when the Streamlit application starts for the first time.

2.  **Search Phase:** The `image_search_ui.py` script handles this phase. A user provides a query image, either by uploading it or selecting it from a gallery. The application then generates a vector embedding for the query image using the same CLIP model. This query embedding is used to search the Milvus collection for the most similar image vectors. The paths of the resulting images are then used to display the search results to the user.

## 4. Component Breakdown

### `embed_images_to_milvus.py`

This script is the data ingestion pipeline for the application. Its primary responsibilities are:

-   **Initializing the Milvus Client:** Establishes a connection to the Milvus database.
-   **Loading the CLIP Model:** Loads the pre-trained CLIP model for generating image embeddings.
-   **Creating the Milvus Collection:** If the collection does not already exist, it creates a new one with a defined schema to store the image vectors and their corresponding file paths.
-   **Image Processing and Embedding:** It recursively scans the `images_folder/train` directory, and for each image, it:
    1.  Opens and preprocesses the image.
    2.  Generates a 512-dimensional vector embedding using the CLIP model.
    3.  Normalizes the embedding.
-   **Data Insertion:** The generated embeddings and their corresponding image paths are inserted into the Milvus collection in batches for efficiency.

### `image_search_ui.py`

This script provides the user interface for the application using Streamlit. Its main functions are:

-   **Database Initialization:** It ensures that the Milvus database is populated by calling the `initialize_database` function from `embed_images_to_milvus.py` when the application starts.
-   **UI Components:** It creates the interactive components of the web application, such as the radio buttons for search mode selection, the file uploader, the image gallery, and the slider for selecting the number of results.
-   **Query Image Processing:** When a user provides a query image, it generates a vector embedding for that image using the CLIP model.
-   **Milvus Search:** It sends the query embedding to the Milvus database to perform a similarity search.
-   **Displaying Results:** The search results, which include the most similar images and their similarity scores, are displayed in the user interface.

## 5. Tunable Parameters

The following parameters can be configured to change the application's behavior:

-   **`MODEL_NAME`:** The CLIP model to be used for generating embeddings (e.g., "ViT-B/32").
-   **`COLLECTION_NAME`:** The name of the Milvus collection where the image embeddings are stored.
-   **`IMAGES_FOLDER`:** The root directory for the training images.
-   **`MILVUS_URI`:** The URI for the Milvus database. It can be a local file path or a remote server address.
-   **`TEST_FOLDER`:** The directory for the test images that appear in the gallery.
-   **`num_results`:** The number of similar images to retrieve and display, adjustable via a slider in the UI.
