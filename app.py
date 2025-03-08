from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv

# Import our custom modules
from document_processing import process_financial_documents, summarize_document
from market_data import get_market_data, get_financial_news
from mother_model import generate_stock_recommendations, simulate_trade

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS to allow requests from frontend
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/api/documents/process', methods=['POST'])
def process_documents():
    """Process and summarize financial documents."""
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Process the document
        document_type = request.form.get('document_type', 'general')
        result = process_financial_documents(file, document_type)
        
        return jsonify({"data": result}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Keep the old endpoint for backward compatibility
@app.route('/api/document/process', methods=['POST'])
def process_documents_legacy():
    """Legacy endpoint for document processing (redirects to the new endpoint)."""
    return process_documents()

@app.route('/api/market/data', methods=['GET'])
def market_data():
    """Get real-time market data for specified symbols."""
    try:
        symbols = request.args.get('symbols', 'AAPL,MSFT,GOOGL').split(',')
        timeframe = request.args.get('timeframe', '1d')
        
        data = get_market_data(symbols, timeframe)
        
        return jsonify({
            "success": True,
            "data": data
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/news', methods=['GET'])
def financial_news():
    """Get latest financial news."""
    try:
        symbols = request.args.get('symbols', None)
        limit = int(request.args.get('limit', 10))
        
        print(f"Received request for news with symbols: {symbols}, limit: {limit}")
        
        news = get_financial_news(symbols, limit)
        
        # Check if news is a list (expected format after our changes)
        if isinstance(news, list):
            return jsonify(news), 200
        # Check if news is a dict with error message
        elif isinstance(news, dict) and "error" in news:
            return jsonify({
                "success": False,
                "error": news["error"],
                "data": news.get("data", [])
            }), 400
        # Fallback for any other format
        else:
            return jsonify(news), 200
    
    except Exception as e:
        print(f"Error in financial_news endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "data": []
        }), 400

@app.route('/api/recommendations', methods=['GET'])
def stock_recommendations():
    """Get AI-driven stock recommendations."""
    try:
        symbols = request.args.get('symbols', 'AAPL,MSFT,GOOGL').split(',')
        timeframe = request.args.get('timeframe', '1m')
        
        recommendations = generate_stock_recommendations(symbols, timeframe)
        
        return jsonify({
            "success": True,
            "data": recommendations
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulate/trade', methods=['POST'])
def trade_simulation():
    """Simulate a trade based on provided parameters."""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ['symbol', 'action', 'amount', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        result = simulate_trade(
            symbol=data['symbol'],
            action=data['action'],
            amount=data['amount'],
            price=data['price'],
            timeframe=data.get('timeframe', '1m')
        )
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Add a message to confirm the server is starting
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"API will be available at http://0.0.0.0:{port}/api")
    print(f"Environment variables loaded: GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY') is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug) 
