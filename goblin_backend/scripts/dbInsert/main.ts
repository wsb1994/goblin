import { Database } from "@db/sqlite";

// Input format based on the provided example
interface ModelOutput {
  id: string;
  timestamp: string;
  comment: string;
}

interface InputData {
  success: boolean;
  model: string;
  is_hate_speech: boolean;
  output: ModelOutput;
}

// Response format
interface OutputResponse {
  success: boolean;
  inserted_records?: number;
  error?: string;
}

function standardOutput(output: string) {
  console.log(output);
}

function initializeDatabase(db: Database) {
  // Create table if it doesn't exist with auto-migration
  db.exec(`
    CREATE TABLE IF NOT EXISTS hate_speech_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      original_id TEXT,
      text TEXT,
      label REAL,
      model TEXT,
      model_is_hate_speech BOOLEAN,
      input_timestamp TEXT
    )
  `);
}

function sanitizeJson(jsonString: string): string {
  try {
    // Remove any leading/trailing whitespace
    let sanitized = jsonString.trim();
    
    // Remove any potential BOM characters
    sanitized = sanitized.replace(/^\uFEFF/, '');
    
    // Try to parse and re-stringify to ensure valid JSON
    const parsed = JSON.parse(sanitized);
    return JSON.stringify(parsed);
  } catch (error) {
    throw new Error(`Invalid JSON format: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function insertResult(db: Database, data: InputData): number {
  let insertedCount = 0;
  
  // Prepare insert statement
  const stmt = db.prepare(`
    INSERT INTO hate_speech_results (
      original_id,
      text,
      label,
      model,
      model_is_hate_speech,
      input_timestamp
    ) VALUES (?, ?, ?, ?, ?, ?)
  `);
  
  try {
    // Insert result only if success is true
    if (data.success && data.output) {
      stmt.run(
        data.output.id || null,
        data.output.comment || null,
        null, // No label in the provided format
        data.model || "unknown",
        data.is_hate_speech ? 1 : 0,
        data.output.timestamp || null
      );
      insertedCount++;
    }
  } finally {
    stmt.finalize();
  }
  
  return insertedCount;
}

export async function processJsonArray(jsonInputs: string[]): Promise<OutputResponse> {
  try {
    // Initialize database connection to current directory
    const db = new Database("./hate_speech_results.db");
    
    try {
      // Initialize database schema (auto-migration)
      initializeDatabase(db);
      
      let totalInserted = 0;
      
      // Process each JSON input serially
      for (const jsonInput of jsonInputs) {
        try {
          // Sanitize the JSON
          const sanitizedJson = sanitizeJson(jsonInput);
          const data: InputData = JSON.parse(sanitizedJson);
          
          // Validate input structure
          if (typeof data.success !== 'boolean') {
            throw new Error("Missing or invalid 'success' field");
          }
          
          if (!data.model) {
            throw new Error("Missing 'model' field");
          }
          
          if (typeof data.is_hate_speech !== 'boolean') {
            throw new Error("Missing or invalid 'is_hate_speech' field");
          }
          
          if (!data.output || !data.output.id || !data.output.comment) {
            throw new Error("Missing or invalid 'output' field with required 'id' and 'comment'");
          }
          
          // Insert the result
          const inserted = insertResult(db, data);
          totalInserted += inserted;
          
        } catch (error) {
          // Log individual JSON processing errors but continue with others
          standardOutput(`Error processing JSON: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
      
      return {
        success: true,
        inserted_records: totalInserted
      };
    } finally {
      db.close();
    }
  } catch (error: unknown) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

export async function main() {
  const args = Deno.args;
  
  // Print all command line arguments as an array to stdout
  standardOutput(`Command line arguments: ${JSON.stringify(args)}`);
  
  if (args.length === 0) {
    standardOutput(JSON.stringify({
      success: false,
      error: 'Please provide JSON input as arguments',
      inserted_records: 0
    }));
    Deno.exit(1);
  }

  try {
    const result = await processJsonArray(args);
    standardOutput(JSON.stringify(result));
    
    if (result.success) {
      Deno.exit(0);
    } else {
      Deno.exit(1);
    }
  } catch (error) {
    standardOutput(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      inserted_records: 0
    }));
    Deno.exit(1);
  }
}

// Run main if this is the main module
if (import.meta.main) {
  main();
}
