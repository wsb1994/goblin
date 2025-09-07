#!/usr/bin/env python3
import sqlite3
import json
import sys
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class ModelOutput:
    id: str
    timestamp: Optional[str] = None
    comment: Optional[str] = None
    text: Optional[str] = None
    label: Optional[Union[str, float]] = None

@dataclass
class InputData:
    success: bool
    model: str
    is_hate_speech: bool
    output: ModelOutput

@dataclass
class OutputResponse:
    success: bool
    inserted_records: Optional[int] = None
    error: Optional[str] = None

def standard_output(output: str) -> None:
    """Print output to stdout (equivalent to console.log)"""
    print(output)

def initialize_database(db_path: str) -> sqlite3.Connection:
    """Initialize database connection and create table if it doesn't exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist with auto-migration
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hate_speech_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_id TEXT,
            comment TEXT,
            label REAL,
            model TEXT,
            model_is_hate_speech BOOLEAN,
            input_timestamp TEXT
        )
    """)
    
    conn.commit()
    return conn

def sanitize_json(json_string: str) -> str:
    """Sanitize JSON input by removing whitespace and BOM characters"""
    try:
        # Remove any leading/trailing whitespace
        sanitized = json_string.strip()
        
        # Remove any potential BOM characters
        sanitized = sanitized.replace('\ufeff', '')
        
        # Try to parse and re-stringify to ensure valid JSON
        parsed = json.loads(sanitized)
        return json.dumps(parsed)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

def insert_result(conn: sqlite3.Connection, data: InputData) -> int:
    """Insert a single result into the database"""
    inserted_count = 0
    cursor = conn.cursor()
    
    try:
        if data.output:
            # Convert label to number if it's a string
            label_value = None
            if data.output.label is not None:
                if isinstance(data.output.label, str):
                    try:
                        label_value = float(data.output.label)
                    except ValueError:
                        label_value = None
                else:
                    label_value = data.output.label
            
            # Use text field if available, otherwise fall back to comment
            text_content = data.output.text or data.output.comment
            
            cursor.execute("""
                INSERT INTO hate_speech_results (
                    original_id,
                    comment,
                    label,
                    model,
                    model_is_hate_speech,
                    input_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.output.id,
                text_content,
                label_value,
                data.model or "unknown",
                1 if data.is_hate_speech else 0,
                data.output.timestamp or None
            ))
            
            inserted_count += 1
            conn.commit()
    
    except Exception as e:
        conn.rollback()
        raise e
    
    return inserted_count

def process_json_array(json_inputs: List[str]) -> OutputResponse:
    """Process an array of JSON inputs and insert them into the database"""
    try:
        # Initialize database connection to current directory
        db_path = "./hate_speech_results.db"
        conn = initialize_database(db_path)
        
        try:
            total_inserted = 0
            
            # Process each JSON input serially
            for json_input in json_inputs:
                try:
                    # Sanitize the JSON
                    sanitized_json = sanitize_json(json_input)
                    data_dict = json.loads(sanitized_json)
                    
                    # Validate input structure
                    if not isinstance(data_dict.get('success'), bool):
                        raise ValueError("Missing or invalid 'success' field")
                    
                    if not data_dict.get('model'):
                        raise ValueError("Missing 'model' field")
                    
                    if not isinstance(data_dict.get('is_hate_speech'), bool):
                        raise ValueError("Missing or invalid 'is_hate_speech' field")
                    
                    output_data = data_dict.get('output')
                    if not output_data or not output_data.get('id') or (not output_data.get('comment') and not output_data.get('text')):
                        raise ValueError("Missing or invalid 'output' field with required 'id' and 'comment' or 'text'")
                    
                    # Create structured data objects
                    model_output = ModelOutput(
                        id=output_data.get('id'),
                        timestamp=output_data.get('timestamp'),
                        comment=output_data.get('comment'),
                        text=output_data.get('text'),
                        label=output_data.get('label')
                    )
                    
                    input_data = InputData(
                        success=data_dict['success'],
                        model=data_dict['model'],
                        is_hate_speech=data_dict['is_hate_speech'],
                        output=model_output
                    )
                    
                    # Insert the result
                    inserted = insert_result(conn, input_data)
                    total_inserted += inserted
                    
                except Exception as error:
                    # Log individual JSON processing errors but continue with others
                    standard_output(f"Error processing JSON: {str(error)}")
            
            return OutputResponse(
                success=True,
                inserted_records=total_inserted
            )
            
        finally:
            conn.close()
            
    except Exception as error:
        return OutputResponse(
            success=False,
            error=str(error)
        )

def main():
    """Main function - equivalent to TypeScript main()"""
    args = sys.argv[1:]  # Skip the script name
    
    # Print all command line arguments as an array to stdout
    standard_output(f"Command line arguments: {json.dumps(args)}")
    
    if len(args) == 0:
        standard_output(json.dumps({
            "success": False,
            "error": "Please provide JSON input as arguments",
            "inserted_records": 0
        }))
        sys.exit(1)
    
    try:
        result = process_json_array(args)
        
        # Convert dataclass to dict for JSON serialization
        result_dict = {
            "success": result.success,
            "inserted_records": result.inserted_records,
            "error": result.error
        }
        
        # Remove None values
        result_dict = {k: v for k, v in result_dict.items() if v is not None}
        
        standard_output(json.dumps(result_dict))
        
        if result.success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as error:
        standard_output(json.dumps({
            "success": False,
            "error": str(error),
            "inserted_records": 0
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
