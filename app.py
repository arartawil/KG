import streamlit as st
from py2neo import Graph
from pyvis.network import Network
import networkx as nx

# Connect to Neo4j
graph = Graph("bolt://3.235.104.75:7687", auth=("neo4j", "thirds-reading-difference"))


# Function to query the Neo4j database and get the graph data
def get_graph_data():
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS source, m.name AS target, type(r) AS relationship
    LIMIT 100
    """
    results = graph.run(query).data()

    return results


# Function to build a NetworkX graph from Neo4j data
def build_networkx_graph(data):
    G = nx.Graph()

    for record in data:
        source = record["source"]
        target = record["target"]
        relationship = record["relationship"]

        G.add_node(source)
        G.add_node(target)
        G.add_edge(source, target, label=relationship)

    return G


# Function to visualize the graph using Pyvis
def visualize_graph(G):
    net = Network(height="750px", width="100%", notebook=False)
    net.from_nx(G)
    net.show("graph.html")

    # Read the generated graph.html and display it in Streamlit
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    st.components.v1.html(source_code, height=750, width=800, scrolling=True)


# Streamlit app
def main():
    st.title("Neo4j Graph Visualization in Streamlit")

    # Fetch graph data
    data = get_graph_data()

    # Build NetworkX graph
    G = build_networkx_graph(data)

    # Visualize the graph
    visualize_graph(G)


if __name__ == "__main__":
    main()
