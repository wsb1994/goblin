use rusqlite::{Connection, Result as SqliteResult};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::env;
use std::process;

#[derive(Debug, Serialize, Deserialize)]
struct ModelOutput {
    id: String,
    timestamp: Option<String>,
    comment: Option<String>,
    text: Option<String>,
    label: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InputData {
    success: bool,
    model: String,
    is_hate_speech: bool,
    output: ModelOutput,
}

#[derive(Debug, Serialize)]
struct OutputResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    inserted_records: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn standard_output(output: &str) {
    println!("{}", output);
}

fn initialize_database(db_path: &str) -> SqliteResult<Connection> {
    let conn = Connection::open(db_path)?;
    
    // Create table if it doesn't exist with auto-migration
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hate_speech_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_id TEXT,
            comment TEXT,
            label REAL,
            model TEXT,
            model_is_hate_speech BOOLEAN,
            input_timestamp TEXT
        )",
        [],
    )?;
    
    Ok(conn)
}

fn sanitize_json(json_string: &str) -> Result<String, String> {
    // Remove any leading/trailing whitespace
    let sanitized = json_string.trim();
    
    // Remove any potential BOM characters
    let sanitized = sanitized.replace('\u{feff}', "");
    
    // Try to parse and re-stringify to ensure valid JSON
    match serde_json::from_str::<Value>(&sanitized) {
        Ok(parsed) => Ok(serde_json::to_string(&parsed).unwrap()),
        Err(e) => Err(format!("Invalid JSON format: {}", e)),
    }
}

fn insert_result(conn: &Connection, data: &InputData) -> SqliteResult<i32> {
    let mut inserted_count = 0;
    
    // Convert label to number if it's a string
    let label_value: Option<f64> = match &data.output.label {
        Some(Value::String(s)) => s.parse().ok(),
        Some(Value::Number(n)) => n.as_f64(),
        _ => None,
    };
    
    // Use text field if available, otherwise fall back to comment
    let text_content = data.output.text.as_ref()
        .or(data.output.comment.as_ref())
        .map(|s| s.as_str());
    
    conn.execute(
        "INSERT INTO hate_speech_results (
            original_id,
            comment,
            label,
            model,
            model_is_hate_speech,
            input_timestamp
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        [
            &data.output.id as &dyn rusqlite::ToSql,
            &text_content,
            &label_value,
            &data.model,
            &(if data.is_hate_speech { 1 } else { 0 }),
            &data.output.timestamp,
        ],
    )?;
    
    inserted_count += 1;
    Ok(inserted_count)
}

fn process_json_array(json_inputs: Vec<String>) -> OutputResponse {
    // Initialize database connection to current directory
    let db_path = "./hate_speech_results.db";
    
    let conn = match initialize_database(db_path) {
        Ok(conn) => conn,
        Err(e) => {
            return OutputResponse {
                success: false,
                inserted_records: None,
                error: Some(format!("Database initialization error: {}", e)),
            };
        }
    };
    
    let mut total_inserted = 0;
    
    // Process each JSON input serially
    for json_input in json_inputs {
        match process_single_json(&conn, &json_input) {
            Ok(inserted) => total_inserted += inserted,
            Err(error) => {
                // Log individual JSON processing errors but continue with others
                standard_output(&format!("Error processing JSON: {}", error));
            }
        }
    }
    
    OutputResponse {
        success: true,
        inserted_records: Some(total_inserted),
        error: None,
    }
}

fn process_single_json(conn: &Connection, json_input: &str) -> Result<i32, String> {
    // Sanitize the JSON
    let sanitized_json = sanitize_json(json_input)?;
    
    // Parse JSON
    let data_dict: Value = serde_json::from_str(&sanitized_json)
        .map_err(|e| format!("JSON parsing error: {}", e))?;
    
    // Validate input structure
    let success = data_dict.get("success")
        .and_then(|v| v.as_bool())
        .ok_or("Missing or invalid 'success' field")?;
    
    let model = data_dict.get("model")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'model' field")?
        .to_string();
    
    let is_hate_speech = data_dict.get("is_hate_speech")
        .and_then(|v| v.as_bool())
        .ok_or("Missing or invalid 'is_hate_speech' field")?;
    
    let output_data = data_dict.get("output")
        .ok_or("Missing 'output' field")?;
    
    let output_id = output_data.get("id")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'id' in output")?
        .to_string();
    
    // Check if either comment or text exists
    let has_comment = output_data.get("comment").and_then(|v| v.as_str()).is_some();
    let has_text = output_data.get("text").and_then(|v| v.as_str()).is_some();
    
    if !has_comment && !has_text {
        return Err("Missing 'comment' or 'text' in output".to_string());
    }
    
    // Create structured data objects
    let model_output = ModelOutput {
        id: output_id,
        timestamp: output_data.get("timestamp").and_then(|v| v.as_str()).map(|s| s.to_string()),
        comment: output_data.get("comment").and_then(|v| v.as_str()).map(|s| s.to_string()),
        text: output_data.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()),
        label: output_data.get("label").cloned(),
    };
    
    let input_data = InputData {
        success,
        model,
        is_hate_speech,
        output: model_output,
    };
    
    // Insert the result
    insert_result(conn, &input_data)
        .map_err(|e| format!("Database insertion error: {}", e))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect(); // Skip the program name
    
    // Print all command line arguments as an array to stdout
    let args_json = serde_json::to_string(&args).unwrap();
    standard_output(&format!("Command line arguments: {}", args_json));
    
    if args.is_empty() {
        let error_response = json!({
            "success": false,
            "error": "Please provide JSON input as arguments",
            "inserted_records": 0
        });
        standard_output(&serde_json::to_string(&error_response).unwrap());
        process::exit(1);
    }
    
    let result = process_json_array(args);
    
    // Serialize result to JSON
    let result_json = serde_json::to_string(&result).unwrap();
    standard_output(&result_json);
    
    if result.success {
        process::exit(0);
    } else {
        process::exit(1);
    }
}
