# SaHS

SaHS is a Harmonized System (HS) Code Classification and Search Engine tailored for ZATCA (Saudi Arabia) customs procedures. It aims to streamline cross-border trade, logistics, and customs clearance by providing accurate, multilingual HS Code lookups and automated classification of commercial invoices.

## Features

- **Semantic Search Engine**: Fast and accurate HS code semantic search across multiple languages using ChromaDB vector store.
- **Multilingual Support**: Supports searching by English, Arabic, and German item descriptions. Automatically translates and aligns terminology.
- **AI Classification**: Leverages Gemini 2.5 Flash to intelligently classify commercial invoice items to the correct Saudi HS code, returning both English and Arabic descriptions alongside ZATCA procedures.
- **Customs & Tariffs Insight**: Displays hierarchy paths, duty rates, and required ZATCA procedures for each classified item.
- **Batch Processing**: Allows uploading CSV invoices for automated mapping of all items at once.
- **Open API Access**: A public, rate-limited API for developers to integrate SaHS classification into their own systems.

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: React / Vite / TypeScript
- **Database/Vector Store**: ChromaDB
- **LLM/Embeddings**: Google Gemini API (`gemini-2.5-flash` and `gemini-embedding-001`)

## Setup

### Prerequisites

- Python 3.10+
- Node.js & npm
- A Google Gemini API Key
- `Tariff.xlsx`: The main customs dataset. You must download this from the [ZATCA Tariff Search Page](https://eservices.zatca.gov.sa/sites/sc/en/tariff/Pages/TariffPages/TariffSearch.aspx) and place it in the project root.

### Backend Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Export your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
3. Initialize the Vector Database (Requires dataset in `Tariff.xlsx`):
   ```bash
   python ingest.py
   ```
4. Start the FastAPI server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Developer Tools & API

SaHS now provides an **Open API** for external integrations.

- **Production API**: `https://sahs-c7oy.onrender.com:8000`
- **Interactive Documentation**:
  - [Swagger UI](https://sahs-c7oy.onrender.com:8000/docs)
  - [ReDoc](https://sahs-c7oy.onrender.com:8000/redoc)
- **API Guide**: See [README_API.md](README_API.md) for detailed implementation instructions and cURL examples.

## Usage

- Navigate to the frontend URL to use the interactive application.
- Use the Search bar to semantically find HS Codes.
- Upload an Invoice CSV to map the entire document to the correct HS Codes.

## License
This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE).
