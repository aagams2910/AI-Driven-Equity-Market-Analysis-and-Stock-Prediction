## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root of the backend directory based on the `.env.example` file:
   ```
   cp .env.example .env
   ```
   
5. Edit the `.env` file and add your API keys and configuration.

## Development

To start the development server:

```
python app.py
```

This will run the Flask application in development mode on http://localhost:5001.

## API Endpoints

### Health Check
- `GET /api/health` - Check if the API is running

### Document Processing
- `POST /api/document/process` - Process and analyze financial documents

### Market Data
- `GET /api/market/data` - Get real-time market data for specified symbols
- `GET /api/news` - Get latest news for specified symbols

### Stock Recommendations
- `GET /api/recommendations` - Get AI-driven stock recommendations

### Trade Simulation
- `POST /api/simulate/trade` - Simulate a trade and analyze potential outcomes

## Project Structure

```
backend/
├── app.py                # Main application entry point
├── document_processing.py # Document processing module
├── market_data.py        # Market data retrieval module
├── mother_model.py       # AI model for predictions
├── utils/                # Utility functions
├── requirements.txt      # Python dependencies
└── .env.example          # Example environment variables
```

## Technologies Used

- Python - Backend language
- Flask - Web framework
- Pandas - Data manipulation
- NumPy - Numerical computations
- PyTorch - Deep learning
- scikit-learn - Machine learning
- NLTK - Natural language processing

## Best Practices

- Follow PEP 8 style guidelines
- Write docstrings for all functions and classes
- Use environment variables for configuration
- Implement proper error handling
- Log important events and errors
