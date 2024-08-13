import streamlit as st
from py2neo import Graph
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Connect to Neo4j
graph = Graph("bolt://3.235.104.75:7687", auth=("neo4j", "thirds-reading-difference"))

# Initialize the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute text embeddings using the pre-trained model
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Function to update the Neo4j database with abstract embeddings
def update_abstract_embeddings():
    query = """
    MATCH (p:Paper)
    WHERE p.embedding IS NULL
    RETURN p.title AS title, p.abstract AS abstract
    """
    papers = graph.run(query).data()

    for paper in papers:
        title = paper['title']
        abstract = paper['abstract']

        if abstract:
            # Compute embeddings for the abstract
            abstract_embedding = get_embeddings(abstract)

            # Update the paper node with the computed embedding
            graph.run("""
            MATCH (p:Paper {title: $title})
            SET p.embedding = $embedding
            """, title=title, embedding=abstract_embedding.tolist())

    st.success("Abstract embeddings have been updated in the Neo4j database.")

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to query the KG by comparing input embeddings with abstract embeddings
def query_kg_by_abstract(input_embeddings, threshold=0.9):
    query = """
    MATCH (p:Paper)
    RETURN p.title AS title, p.abstract AS abstract, p.embedding AS abstract_embedding, 
           [(p)<-[:AUTHORED]-(a:Author) | a.name] AS authors
    """
    papers = graph.run(query).data()

    matching_papers = []

    for paper in papers:
        abstract_embedding = paper['abstract_embedding']

        # Skip if the abstract_embedding is None
        if abstract_embedding is None:
            continue

        # Calculate similarity with the abstract
        abstract_similarity = cosine_similarity(input_embeddings, np.array(abstract_embedding))

        if abstract_similarity >= threshold:
            matching_papers.append({
                'title': paper['title'],
                'abstract': paper['abstract'],
                'similarity': abstract_similarity,
                'authors': paper['authors']
            })

    # Sort papers by abstract similarity
    matching_papers = sorted(matching_papers, key=lambda x: x['similarity'], reverse=True)

    return matching_papers

# Display results in Streamlit
def display_results(results):
    if not results:
        st.write("No matching papers found.")
    else:
        for result in results:
            st.write(f"### Title: {result['title']}")
            st.write(f"**Abstract:** {result['abstract']}")
            st.write(f"**Similarity:** {result['similarity']:.2f}")
            st.write(f"**Authors:** {', '.join(result['authors'])}")
            st.write("---")

# Streamlit App
def main():
    st.title("Academic Research Assistant")
    st.write("Enter a description of your research idea, and find the most relevant research papers and authors.")

    if st.button("Update Abstract Embeddings"):
        # Update abstract embeddings in the database
        update_abstract_embeddings()

    user_input = st.text_area("Enter your research idea:")

    # Input Constraints
    min_length = 20
    required_keywords = ["machine learning", "neural networks"]

    if st.button("Find Relevant Research"):
        if not user_input:
            st.write("Please enter a research idea to start the search.")
        elif len(user_input) < min_length:
            st.write(f"Your input must be at least {min_length} characters long.")
        elif not any(keyword in user_input.lower() for keyword in required_keywords):
            st.write(f"Your input must include one of the following keywords: {', '.join(required_keywords)}")
        else:
            st.write("Analyzing your input...")
            input_embeddings = get_embeddings(user_input)

            st.write("Searching for relevant papers...")
            results = query_kg_by_abstract(input_embeddings)
            display_results(results)

if __name__ == "__main__":
    main()
