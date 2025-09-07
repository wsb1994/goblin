import { Database } from "@db/sqlite";

// Input format: {id, text, label} or {id, comment, label}
interface InputData {
  id?: string;
  text?: string;
  comment?: string;
  label?: number;
  [key: string]: any;
}

// Output format for each model
interface ModelResult {
  success: boolean;
  is_hate_speech?: boolean;
  model?: string;
  output?: InputData;
  error?: string;
}

// Expected input structure
interface DbInsertInput {
  ModelA: ModelResult;
  ModelB: ModelResult;
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
      model_is_hate_speech BOOLEAN
    )
  `);
}

function insertResults(db: Database, model: ModelResult): number {
  let insertedCount = 0;
  
  // Prepare insert statement
  const stmt = db.prepare(`
    INSERT INTO hate_speech_results (
      original_id,
      text,
      label,
      model,
      model_is_hate_speech
    ) VALUES (?, ?, ?, ?, ?)
  `);
  
  try {

    // Insert ModelB result only if success is true
    if (model.output) {
      const data = model.output;
      stmt.run(
        data.id || null,
        data.text || data.comment || null,
        data.label || null,
        modelB.model || "ModelB",
        modelB.is_hate_speech ? 1 : 0
      );
      insertedCount++;
    }
  } finally {
    stmt.finalize();
  }
  
  return insertedCount;
}

export async function processJson(jsonInput: string): Promise<OutputResponse> {
  try {
    const data: DbInsertInput = JSON.parse(jsonInput);
    
    // Validate input structure
    if (!data.ModelA || !data.ModelB) {
      return {
        success: false,
        error: "Missing ModelA or ModelB results in input data"
      };
    }

    // Initialize database connection to current directory
    const db = new Database("./hate_speech_results.db");
    
    try {
      // Initialize database schema (auto-migration)
      initializeDatabase(db);
      
      // Insert the results (only successful ones)
      let insertedCount = insertResults(db, data.ModelA);
      insertedCount += insertResults(db, data.ModelB);
      
      return {
        success: true,
        inserted_records: insertedCount
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
  
  if (args.length === 0) {
    standardOutput(JSON.stringify({
      success: false,
      error: 'Please provide JSON input as argument',
      inserted_records: 0
    }));
    Deno.exit(1);
  }

  try {
    const result = await processJson(args[0]);
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
