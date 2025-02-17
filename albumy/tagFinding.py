from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch


def classify_local_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Load the processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Process the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

#
# # Example usage
# image_path = "path/to/your/local/image.jpg"  # Replace with your image path
# classify_local_image(image_path)