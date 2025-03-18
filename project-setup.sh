# Create a project directory
mkdir morocco_trip_planner
cd morocco_trip_planner

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install crewai transformers langchain sentence-transformers requests
pip install ollama  # For local model support
