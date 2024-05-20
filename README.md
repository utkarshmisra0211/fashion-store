# Clothes Recommender System

This is a clothes recommender system built using Streamlit, TensorFlow, and Scikit-learn. The application allows users to upload an image of a clothing item, and it recommends similar items from a predefined dataset. The system uses a pre-trained ResNet50 model to extract features from the uploaded image and compares them with features from the dataset using Nearest Neighbors.

## Features

- **Gender Selection**: Users can select the gender for which they want to find clothes.
- **Clothes Selection**: Users can select the type of clothing they are interested in.
- **Image Upload**: Users can upload an image of the clothing item they want recommendations for.
- **Recommendations**: The system recommends similar clothing items based on the uploaded image.

## Requirements

- Python 3.6+
- Streamlit
- TensorFlow
- Scikit-learn
- Pillow
- NumPy

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/clothes-recommender.git
    cd clothes-recommender
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model weights and place them in the appropriate directory** (if necessary).

4. **Ensure the directory structure is as follows**:
    ```
    clothes-recommender/
    ├── data/
    │   └── manyavar_data/
    │       ├── men/
    │       │   ├── indo-western/
    │       │   ├── jacket/
    │       │   ├── kurta-dhoti/
    │       │   ├── kurta-jacket-set/
    │       │   ├── kurta/
    │       │   └── suits/
    │       └── women/
    │           ├── gown/
    │           ├── saree/
    │           └── stitched-suit/
    ├── embedings/
    │   ├── men/
    │   │   ├── indo-west.pkl
    │   │   ├── jacket.pkl
    │   │   ├── kurta-dhoti.pkl
    │   │   ├── kurta-jacket-set.pkl
    │   │   ├── kurta.pkl
    │   │   └── suits.pkl
    │   └── women/
    │       ├── gown.pkl
    │       ├── saree.pkl
    │       └── stitched-suit.pkl
    ├── image_names/
    │   ├── men/
    │   │   ├── indo-west.pkl
    │   │   ├── jacket.pkl
    │   │   ├── kurta-dhoti.pkl
    │   │   ├── kurta-jacket-set.pkl
    │   │   ├── kurta.pkl
    │   │   └── suits.pkl
    │   └── women/
    │       ├── gown.pkl
    │       ├── saree.pkl
    │       └── stitched-suit.pkl
    ├── uploads/
    └── app.py
    ```

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Interact with the application**:
   - Select the gender.
   - Select the type of clothing.
   - Upload an image of the clothing item you want recommendations for.
   - View the recommended clothing items.

## Code Overview

### Model Definition

The model used for feature extraction is a pre-trained ResNet50 with the top layer removed and a GlobalMaxPooling2D layer added.

### Feature Extraction

The `extract_features` function preprocesses the image and extracts features using the model.

### Recommendation

The `recommend` function uses Nearest Neighbors to find similar items.

### Streamlit Application

The Streamlit app handles user input and displays recommendations.

---

Feel free to reach out if you have any questions or need further assistance!
