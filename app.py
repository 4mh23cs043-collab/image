import os
import torch  # type: ignore
from flask import Flask, render_template, request, jsonify  # type: ignore
from torchvision import models, transforms  # type: ignore
from PIL import Image  # type: ignore
import torch.nn as nn  # type: ignore

from classifier_module import get_model # type: ignore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device Configuration
device = torch.device("cpu")

# Load classes
CLASSES = []
if os.path.exists("classes.txt"):
    with open("classes.txt", "r") as f:
        CLASSES = [line.strip() for line in f.readlines()]
else:
    CLASSES = ['Non-Pets', 'Pets'] # Default fallback

# Initialize and Load Models
MODEL_PATH = "classifier_model.pth"

# 1. Specific Identification Model (ImageNet)
weights = models.MobileNet_V2_Weights.DEFAULT
specific_model = models.mobilenet_v2(weights=weights)
specific_model.eval()
specific_model.to(device)
CATEGORIES = weights.meta["categories"]

# 2. Custom Broad Classifier Model (Pets vs Non-Pets)
def load_custom_model():
    model = get_model(len(CLASSES), weights=None)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded custom model weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Using uninitialized classifier.")
    model.eval()
    model.to(device)
    return model

custom_model = load_custom_model()

# Animal metadata: diet type and habitat
ANIMAL_METADATA = {
    # Big Cats
    'lion': {'diet': 'Carnivore', 'habitat': 'African Savannas and Grasslands'},
    'tiger': {'diet': 'Carnivore', 'habitat': 'Asian Forests and Grasslands'},
    'leopard': {'diet': 'Carnivore', 'habitat': 'African and Asian Forests'},
    'cheetah': {'diet': 'Carnivore', 'habitat': 'African Savannas'},
    'jaguar': {'diet': 'Carnivore', 'habitat': 'South American Rainforests'},
    'cougar': {'diet': 'Carnivore', 'habitat': 'American Mountains and Forests'},
    'puma': {'diet': 'Carnivore', 'habitat': 'American Mountains and Forests'},
    'panther': {'diet': 'Carnivore', 'habitat': 'Forests and Swamps'},
    'lynx': {'diet': 'Carnivore', 'habitat': 'Northern Forests'},
    
    # Canines - Wild
    'wolf': {'diet': 'Carnivore', 'habitat': 'Forests and Tundra'},
    'fox': {'diet': 'Omnivore', 'habitat': 'Forests, Grasslands, and Urban Areas'},
    'coyote': {'diet': 'Omnivore', 'habitat': 'North American Prairies and Deserts'},
    
    # Dogs - Common Breeds (all domesticated, omnivores)
    'dog': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'bulldog': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'boxer': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'pug': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'beagle': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'retriever': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'labrador': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'golden': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'shepherd': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'german': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'poodle': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'husky': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'terrier': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'spaniel': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'chihuahua': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'dachshund': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'rottweiler': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'doberman': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'mastiff': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'corgi': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'shiba': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'akita': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'collie': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'pointer': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    'setter': {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'},
    
    # Bears
    'bear': {'diet': 'Omnivore', 'habitat': 'Forests and Mountains'},
    'grizzly': {'diet': 'Omnivore', 'habitat': 'North American Forests'},
    'polar': {'diet': 'Carnivore', 'habitat': 'Arctic Ice and Tundra'},
    
    # Domestic Cats
    'cat': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'tabby': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'siamese': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'persian': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'egyptian': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'maine': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'ragdoll': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    'bengal': {'diet': 'Carnivore', 'habitat': 'Domesticated - Human Homes'},
    
    # Birds
    'bird': {'diet': 'Omnivore', 'habitat': 'Various - Trees, Grasslands, Urban'},
    'parrot': {'diet': 'Herbivore', 'habitat': 'Tropical Forests'},
    'eagle': {'diet': 'Carnivore', 'habitat': 'Mountains and Coastal Areas'},
    'owl': {'diet': 'Carnivore', 'habitat': 'Forests and Grasslands'},
    'robin': {'diet': 'Omnivore', 'habitat': 'Gardens and Woodlands'},
    'sparrow': {'diet': 'Omnivore', 'habitat': 'Urban and Rural Areas'},
    
    # Rabbits
    'rabbit': {'diet': 'Herbivore', 'habitat': 'Grasslands and Forests'},
    'hare': {'diet': 'Herbivore', 'habitat': 'Open Fields and Grasslands'},
    'bunny': {'diet': 'Herbivore', 'habitat': 'Domesticated - Human Homes'},
    
    # Other
    'elephant': {'diet': 'Herbivore', 'habitat': 'African and Asian Savannas'},
    'giraffe': {'diet': 'Herbivore', 'habitat': 'African Savannas'},
    'zebra': {'diet': 'Herbivore', 'habitat': 'African Grasslands'},
    'monkey': {'diet': 'Omnivore', 'habitat': 'Tropical Forests'},
    'gorilla': {'diet': 'Herbivore', 'habitat': 'African Rainforests'},
    'hyena': {'diet': 'Carnivore', 'habitat': 'African Savannas'},
    'horse': {'diet': 'Herbivore', 'habitat': 'Domesticated - Farms and Ranches'},
    'cow': {'diet': 'Herbivore', 'habitat': 'Domesticated - Farms'},
    'pig': {'diet': 'Omnivore', 'habitat': 'Domesticated - Farms'},
    'sheep': {'diet': 'Herbivore', 'habitat': 'Domesticated - Farms'},
}

def get_animal_info(animal_name):
    """Get diet and habitat information for an animal."""
    animal_lower = animal_name.lower()
    
    # Check for exact or partial matches
    for key, info in ANIMAL_METADATA.items():
        if key in animal_lower:
            return info
    
    # Improved fallback: try to infer from common patterns
    # If it contains common pet indicators, assume domestic
    domestic_indicators = ['retriever', 'terrier', 'spaniel', 'hound', 'poodle']
    if any(indicator in animal_lower for indicator in domestic_indicators):
        return {'diet': 'Omnivore', 'habitat': 'Domesticated - Human Homes'}
    
    # Default fallback
    return {'diet': 'Unknown', 'habitat': 'Unknown'}


# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Wild animal keywords for classification override
WILD_ANIMAL_KEYWORDS = [
    'lion', 'tiger', 'leopard', 'cheetah', 'snow leopard', 'jaguar', 'cougar', 'puma', 
    'panther', 'lynx', 'wolf', 'fox', 'bear', 'hyena', 'coyote', 'bobcat', 'jaguarundi'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            image = Image.open(file_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)

            # 1. Specific Identification (Animal Name) - Run first for guardrails
            with torch.no_grad():
                specific_outputs = specific_model(img_tensor)
                _, specific_idx = specific_outputs.max(1)
                specific_name_raw = CATEGORIES[specific_idx.item()]
            
            specific_name = specific_name_raw.replace('_', ' ').title()

            # 2. Broad Classification (Pets vs Non-Pets)
            with torch.no_grad():
                outputs = custom_model(img_tensor)
                _, predicted = outputs.max(1)
                class_idx = predicted.item()
                broad_label = CLASSES[class_idx]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()

            # 3. Guardrail Logic: Override if specific identification is a wild animal
            is_wild = any(keyword in specific_name_raw.lower() for keyword in WILD_ANIMAL_KEYWORDS)
            if is_wild and broad_label == 'Pets':
                broad_label = 'Non-Pets'

            # 4. Get animal metadata (diet and habitat)
            animal_info = get_animal_info(specific_name_raw)

            result = {
                'class': broad_label,
                'specific_name': specific_name,
                'diet': animal_info['diet'],
                'habitat': animal_info['habitat'],
                'confidence': f"{confidence * 100:.2f}%",
                'image_url': file_path.replace('\\', '/')
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
