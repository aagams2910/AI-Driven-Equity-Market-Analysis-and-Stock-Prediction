import os
import tempfile
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
import google.generativeai as genai

# Load API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Define document types and their processors
DOCUMENT_TYPES = {
    "earnings_report": {
        "description": "Quarterly or annual earnings reports",
        "questions": [
            "What are the key financial highlights?",
            "How did the company perform compared to previous periods?",
            "What are the revenue and profit figures?",
            "What are the future projections or guidance?",
            "What are the main risk factors mentioned?"
        ]
    },
    "regulatory_filing": {
        "description": "SEC filings like 10-K, 10-Q, 8-K",
        "questions": [
            "What are the key disclosures in this filing?",
            "Are there any significant changes in financial position?",
            "What are the risk factors mentioned?",
            "Are there any legal proceedings or investigations?",
            "What are the management's discussion and analysis highlights?"
        ]
    },
    "market_analysis": {
        "description": "Market research and analysis reports",
        "questions": [
            "What are the key market trends identified?",
            "What is the competitive landscape described?",
            "What are the growth projections for the market?",
            "What are the main challenges or obstacles mentioned?",
            "What opportunities are highlighted in the report?"
        ]
    },
    "general": {
        "description": "General financial document",
        "questions": [
            "What are the key points in this document?",
            "What financial data is presented?",
            "What are the main conclusions or recommendations?",
            "Are there any risks or challenges mentioned?",
            "What are the future outlook or projections?"
        ]
    }
}

def process_financial_documents(file, document_type: str = "general") -> Dict[str, Any]:
    """
    Process financial documents using Google's Gemini API
    
    Args:
        file: The uploaded file object
        document_type: Type of financial document (earnings_report, regulatory_filing, etc.)
        
    Returns:
        Dict containing processed data and analysis
    """
    if document_type not in DOCUMENT_TYPES:
        document_type = "general"
        
    # Save uploaded file to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    # Determine file type and load document
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        # Extract text from the document
        if file_extension == '.pdf':
            try:
                # For PDF files, use a simple text extraction
                import PyPDF2
                with open(temp_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        else:
                            print(f"Warning: Could not extract text from a page in {file.filename}")
                
                if not text.strip():
                    raise ValueError(f"Could not extract any text from the PDF file. The file may be scanned or contain only images.")
                
                print(f"Successfully extracted {len(text)} characters from PDF")
            except Exception as pdf_error:
                print(f"Error extracting text from PDF: {str(pdf_error)}")
                raise ValueError(f"Failed to process PDF file: {str(pdf_error)}")
        elif file_extension in ['.csv', '.xlsx', '.xls']:
            try:
                if file_extension == '.csv':
                    df = pd.read_csv(temp_path)
                else:
                    # For Excel files, convert to dataframe and then to text
                    df = pd.read_excel(temp_path)
                text = df.to_string()
                print(f"Successfully processed spreadsheet with {len(df)} rows")
            except Exception as excel_error:
                print(f"Error processing spreadsheet: {str(excel_error)}")
                raise ValueError(f"Failed to process spreadsheet file: {str(excel_error)}")
        else:
            # For text files or other formats
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Successfully read text file with {len(text)} characters")
            except UnicodeDecodeError:
                # Try with a different encoding
                try:
                    with open(temp_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                    print(f"Successfully read text file with latin-1 encoding, {len(text)} characters")
                except Exception as text_error:
                    print(f"Error reading text file: {str(text_error)}")
                    raise ValueError(f"Failed to read text file: {str(text_error)}")
            except Exception as text_error:
                print(f"Error reading text file: {str(text_error)}")
                raise ValueError(f"Failed to read text file: {str(text_error)}")
        
        # Check if we have enough text to process
        if len(text.strip()) < 100:
            raise ValueError("The document contains too little text to analyze (less than 100 characters).")
        
        # Process document with specific questions based on document type
        results = {}
        
        # Initialize Gemini 1.5 Flash model
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        
        # Get questions based on document type
        questions = DOCUMENT_TYPES.get(document_type, DOCUMENT_TYPES["general"])["questions"]
        
        # Process each question
        for question in questions:
            prompt = f"""
            Based on the following document, please answer this question: {question}
            
            Document content:
            {text[:50000]}  # Increased text limit for Gemini 1.5 Flash
            
            Provide a detailed, factual answer based only on the information in the document.
            Include specific numbers, percentages, and metrics where relevant.
            """
            
            response = model.generate_content(prompt)
            results[question] = response.text
            
        # Generate summary
        try:
            summary = summarize_document(results)
        except Exception as summary_error:
            print(f"Error generating summary: {str(summary_error)}")
            # Provide a default summary if summarization fails
            summary = {
                "executive_summary": "Could not generate executive summary due to an error.",
                "key_metrics": "Could not extract key metrics due to an error.",
                "risks": "Could not analyze risks due to an error.",
                "opportunities": "Could not identify opportunities due to an error.",
                "recommendations": "Could not generate recommendations due to an error."
            }
        
        return {
            "document_type": document_type,
            "filename": file.filename,
            "analysis": results,
            "summary": summary,
            "status": "success"
        }
        
    except Exception as e:
        # Log the error
        print(f"Error processing document {file.filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Determine a more user-friendly error message
        error_message = str(e)
        if "PDF file" in error_message:
            user_message = "Could not process this PDF file. It may be encrypted, scanned, or contain only images."
        elif "spreadsheet" in error_message:
            user_message = "Could not process this spreadsheet file. Please check the file format and try again."
        elif "text file" in error_message:
            user_message = "Could not read this text file. Please check the file encoding and try again."
        elif "too little text" in error_message:
            user_message = "The document contains too little text to analyze. Please upload a document with more content."
        elif "API key" in error_message or "authentication" in error_message.lower():
            user_message = "API authentication error. Please check your API key configuration."
        else:
            user_message = "An error occurred while processing the document. Please try again with a different file."
        
        # Ensure we return a valid summary object even in case of errors
        return {
            "status": "error",
            "error": user_message,
            "technical_error": str(e),
            "document_type": document_type,
            "filename": file.filename,
            "summary": {
                "executive_summary": user_message,
                "key_metrics": "Not available due to processing error.",
                "risks": "Not available due to processing error.",
                "opportunities": "Not available due to processing error.",
                "recommendations": "Not available due to processing error."
            }
        }
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def summarize_document(processed_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a summary of the processed document data using Gemini 1.5 Flash
    
    Args:
        processed_data: Dictionary containing processed document data
        
    Returns:
        Dictionary with summary sections
    """
    # Initialize Gemini 1.5 Flash model for summarization
    generation_config = {
        "temperature": 0.1,  # Lower temperature for more factual responses
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,  # Increased token limit for more detailed summaries
    }
    
    # Use Gemini 1.5 Flash model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    # Combine all the processed data into a single text
    combined_text = ""
    for question, answer in processed_data.items():
        combined_text += f"Question: {question}\nAnswer: {answer}\n\n"
    
    # Create a comprehensive prompt for document summarization
    summary_prompt = f"""
    You are a financial analyst tasked with summarizing a financial document. 
    Below is the extracted information from the document. Please analyze this information 
    and provide a comprehensive summary with the following sections:

    1. EXECUTIVE SUMMARY: Provide a concise 3-4 paragraph summary of the key points from the document.
    
    2. KEY METRICS: Extract and list all important financial metrics and KPIs mentioned in the document. 
       Include exact figures, percentages, growth rates, and time periods. Format as bullet points.
       Examples of metrics to look for:
       - Revenue and revenue growth
       - Profit margins (gross, operating, net)
       - EBITDA and EBITDA margins
       - EPS (Earnings Per Share)
       - Cash flow figures
       - ROI, ROE, ROCE
       - Debt ratios and leverage
       - Market share percentages
       - Customer acquisition costs
       - User/customer growth
    
    3. RISKS: Identify and list the main risks, challenges, or negative factors mentioned in the document.
    
    4. OPPORTUNITIES: Identify and list the main opportunities, positive developments, or growth areas mentioned.
    
    5. RECOMMENDATIONS: Based on the information provided, what are 3-5 key recommendations or takeaways?

    Document information:
    {combined_text}

    Respond with ONLY these five sections, formatted clearly. Be specific, factual, and concise.
    """
    
    # Generate the summary
    response = model.generate_content(summary_prompt)
    summary_text = response.text
    
    # Parse the response to extract each section
    sections = {
        "executive_summary": "",
        "key_metrics": "",
        "risks": "",
        "opportunities": "",
        "recommendations": ""
    }
    
    # Simple parsing logic to extract sections
    current_section = None
    for line in summary_text.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for section headers
        if "EXECUTIVE SUMMARY" in line.upper() or "1." in line and "EXECUTIVE SUMMARY" in line.upper():
            current_section = "executive_summary"
            continue
        elif "KEY METRICS" in line.upper() or "2." in line and "KEY METRICS" in line.upper():
            current_section = "key_metrics"
            continue
        elif "RISKS" in line.upper() or "3." in line and "RISKS" in line.upper():
            current_section = "risks"
            continue
        elif "OPPORTUNITIES" in line.upper() or "4." in line and "OPPORTUNITIES" in line.upper():
            current_section = "opportunities"
            continue
        elif "RECOMMENDATIONS" in line.upper() or "5." in line and "RECOMMENDATIONS" in line.upper():
            current_section = "recommendations"
            continue
            
        # Add content to the current section
        if current_section and current_section in sections:
            if sections[current_section]:
                sections[current_section] += "\n" + line
            else:
                sections[current_section] = line
    
    # If any section is empty, add a default message
    for section in sections:
        if not sections[section]:
            sections[section] = f"No {section.replace('_', ' ')} information found in the document."
    
    return sections 
