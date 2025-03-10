import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from PIL import Image

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Load a text embedding model for RAG
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Embedding dimension for MiniLM
index = faiss.IndexFlatL2(dimension)

documents = [
    {"text": "The Eiffel Tower is located in Paris, France.", "image": "C:\\Users\\prafu\\OneDrive\\Documents\\eiffel.jpg"},
    {"text": "The robot.", "image": "C:\\Users\\prafu\\OneDrive\\Documents\\robot.jpg"}
]

# Indexing the documents
for doc in documents:
    embedding = embedder.encode(doc["text"])
    index.add(np.array([embedding]))


def query_rag(image_path, text_query):
    # Process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Generate image caption
    with torch.no_grad():
        caption_ids = model.generate(**inputs)
        caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]

    # Combine image caption and text query for RAG
    combined_query = caption + " " + text_query
    query_embedding = embedder.encode(combined_query)

    # Search FAISS index
    _, I = index.search(np.array([query_embedding]), k=1)
    matched_doc = documents[I[0][0]]

    # Generate a detailed response with the retrieved text
    detailed_text = matched_doc["text"] + "\n" + text_query
    detailed_input = processor(text=detailed_text, images=image, return_tensors="pt")

    with torch.no_grad():
        detailed_ids = model.generate(pixel_values=detailed_input.pixel_values, input_ids=detailed_input.input_ids, attention_mask=detailed_input.attention_mask)
        detailed_response = processor.batch_decode(detailed_ids, skip_special_tokens=True)[0]

    return {
        "image_caption": caption,
        "retrieved_text": matched_doc["text"],
        "detailed_response": detailed_response
    }


if __name__ == "__main__":
    image_path = "C:\\Users\\prafu\\OneDrive\\Documents\\robot.jpg"
    text_query = "Where is this place?"
    result = query_rag(image_path, text_query)
    
    print("Image Caption:", result["image_caption"])
    print("Retrieved Text:", result["retrieved_text"])
    print("Detailed Response:", result["detailed_response"])

