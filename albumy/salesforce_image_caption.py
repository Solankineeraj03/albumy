import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def generate_image_captions(image_path, conditional_text="a photography of"):
    """
    Generates both conditional and unconditional captions for a given image.

    :param image_path: Path to the image file.
    :param conditional_text: Text prompt for conditional captioning (default: "a photography of").
    :return: Tuple containing conditional and unconditional captions.
    """
    # Load BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load local image
    raw_image = Image.open(image_path).convert('RGB')

    # Conditional image captioning
    inputs = processor(raw_image, conditional_text, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

    return conditional_caption, unconditional_caption


# Example usage
# image_path = "uploads/imageName34.jpg"  # Change this to your image file path
# cond_caption, uncond_caption = generate_image_captions(image_path)
# print("Conditional Caption:", cond_caption)
# print("Unconditional Caption:", uncond_caption)
