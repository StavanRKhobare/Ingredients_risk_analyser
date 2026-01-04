# Food Ingredient Risk Classifier

An AI-powered application that classifies food ingredients into risk categories and provides concise safety explanations. This tool helps consumers make informed decisions about the products they buy by analyzing ingredient lists and identifying potential health concerns.

## Features

- **Ingredient Risk Classification**: Classifies food ingredients into 5 risk levels (1-5) using a fine-tuned DeBERTa model
- **Concise Safety Explanations**: Provides clear, concise explanations for each ingredient's safety profile
- **Web Interface**: User-friendly Streamlit interface for easy interaction
- **API Access**: RESTful API for programmatic access to the classification engine

## Risk Levels

- 🟢 **Level 1-2**: Very Safe/Safe - Natural ingredients
- 🟡 **Level 3**: Moderate - Refined but generally safe
- 🟠 **Level 4**: Concerning - Artificial additives
- 🔴 **Level 5**: High Risk - Potentially harmful substances

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **AI Models**: 
  - DeBERTa transformer for ingredient risk classification
  - Groq LLM API for generating explanations
- **Vector Storage**: FAISS
- **Embeddings**: Sentence Transformers

## Prerequisites

- Python 3.8+
- Groq API key (set in `.env` file)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bt-el
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Running the Backend Server

```bash
uvicorn backend:app --reload
```

The backend API will be available at `http://localhost:8000`.

### Running the Frontend Application

```bash
streamlit run frontend.py
```

The web interface will be available at `http://localhost:8501`.

## API Endpoints

- `POST /predict` - Classify ingredients and get risk assessment
- `GET /health` - Health check endpoint
- `GET /` - API information

### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "refined wheat flour, sugar, edible vegetable oil (palmolein), emulsifier (322), synthetic food colour (INS 133)"}'
```

## Project Structure

```
.
├── backend.py              # FastAPI backend server
├── frontend.py             # Streamlit web interface
├── rag_pipeline.py         # LLM explanation generation pipeline
├── dataset_documentation.md # Ingredient safety documentation
├── requirements.txt        # Python dependencies
└── .env                   # Environment variables (not included in repo)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.
