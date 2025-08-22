!pip install -q sentence-transformers faiss-cpu PyMuPDF transformers accelerate

import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.colab import files
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Please upload your PDF file:")
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
print(f"Uploaded file: {pdf_path}")

doc = fitz.open(pdf_path)
text = ""
for page in doc:

    text += page.get_text()
print(f"Extracted {len(text)} characters from PDF")

def split_text(text, max_length=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += " " + sentence
        else:
            if chunk:
                chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

chunks = split_text(text, max_length=500)
print(f"Split text into {len(chunks)} chunks")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Creating embeddings for chunks...")
embeddings = embedder.encode(chunks, convert_to_tensor=False)
embedding_vectors = np.array(embeddings).astype('float32')
embedding_vectors = embedding_vectors / np.linalg.norm(embedding_vectors, axis=1, keepdims=True)

dimension = embedding_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embedding_vectors)
print(f"FAISS index built with {index.ntotal} vectors (cosine similarity)")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def answer_question_local(question, k=5, max_new_tokens=200, threshold=0.3, show_context=True):
    q_emb = embedder.encode([question], convert_to_tensor=False)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = np.array(q_emb).astype('float32')
    similarities, indices = index.search(q_emb, k)
    if similarities[0][0] < threshold:
        return "Answer not found in document."
    context_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(context_chunks)
    if show_context:
        print("\n--- Retrieved Context ---")
        for i, chunk in enumerate(context_chunks):
            print(f"[{i+1}] {chunk[:300]}...\n")
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nQ: What is supervised learning?")
print("A:", answer_question_local("What is supervised learning?"))

print("\nQ: Who is the president of the USA?")
print("A:", answer_question_local("Who is the president of the USA?"))
