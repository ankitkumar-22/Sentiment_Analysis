1. **Clone the repository:**
   ```bash
   git clone https://github.com/masterK0927/newsAnalysis.git
   cd news-analysis
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Environment Setup: Create a .env file with the following variables:
   ```bash
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your_pinecone_index_name
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```