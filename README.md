# Image Classification: Pets vs Non-Pets

An end-to-end image classification system built with PyTorch and Flask.

## Features
- **Classification**: Classifies images into 'Pets' (Dogs, Cats, Birds, Rabbits) and 'Non-Pets'.
- **Specific Identification**: Uses a pre-trained ImageNet model to identify the specific breed or animal name.
- **Diet & Habitat**: Provides the diet type (Herbivore, Carnivore, Omnivore) and natural habitat for identified animals.
- **Wild Animal Guardrails**: Specialized logic to correctly classify wild felines (Lions, Tigers, etc.) as Non-Pets even if they share visual features with domestic cats.
- **Modern UI**: A premium, responsive web interface for easy image uploads and result viewing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/4mh23cs043-collab/image.git
   cd image
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision flask pillow
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and visit: `http://127.0.0.1:5001`

## Model
The system uses a custom-trained **MobileNetV2** model for broad category classification and a default ImageNet **MobileNetV2** for fine-grained animal identification.
