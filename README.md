# xstream-gnn-modular-design
GNN-based environmental intelligence system for modular architectural design in extreme climates

Xstream: GNN-based Environmental Design Intelligence
Overview

Xstream is an AI-driven design system developed for architects and engineers working in extreme environments.

It uses Graph Neural Networks (GNNs) to learn relationships between:

Modular building geometry
Site-specific environmental conditions

The system enables real-time evaluation of environmental and spatial performance during early-stage design.

🧠 Core Idea

Architectural systems are converted into graphs:

Nodes → Modules
Edges → Connections between modules

Each node encodes:

Geometric properties (area, volume, bounding box)
Climate data (wind, radiation, temperature)
⚙️ Features
Environmental prediction:
Solar radiation
Wind exposure
Energy consumption
Design evaluation:
Visibility percentage
Egress distance
Data-driven design suggestions from module catalogue
📊 System Workflow
Modular design generated in Grasshopper
Geometry converted into graph structure
Climate data integrated into node features
GNN model processes graph
Environmental + design metrics predicted
🖼️ Visual Representation
Graph Structure

Site Distribution

🛠️ Tech Stack
Python
Graph Neural Networks
NetworkX
Grasshopper (Rhino)

▶️ How to Run
pip install -r requirements.txt

Run notebook:

notebooks/gnn_model.ipynb
📌 Future Work
Integration with live Grasshopper plugin
Real-time feedback loop
Expanded dataset for training
