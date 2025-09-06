import * as sqlite from "jsr:@db/sqlite";


/*
{id, text, label}
label is 1.0 for hate speech, claude should never know this, but this is your input.


output is {
success: boolean,
is_hate_speech: boolean,
model: "ModelA",
output: {id, text, label} // echo input
error?: string // if success is false
}

for both ModelA and ModelB disregard labels that fail, if success: boolean is false don't do anything and disregard the insert.
for all other inputs (labeled output above) insert into sqlite db with:
timestamp
id from input
text from input
label from input
model: "ModelA" or "ModelB"
model_is_hate_speech: boolean
*/


function standardOutput(output: string) {
  console.log(output);
}

interface ModelResult {
  success: boolean;
  is_hate_speech?: boolean;
  model?: string;
  output?: any;
  error?: string;
}

interface DbInsertInput {
  ModelA: ModelResult;
  ModelB: ModelResult;
}

interface OutputResponse {
  success: boolean;
  inserted_records?: number;
  error?: string;
}

async function initializeDatabase(db: sqlite.Database) {
  // Create table if it doesn't exist
  db.exec(`
    CREATE TABLE IF NOT EXISTS hate_speech_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      original_data TEXT,
      model_a_name TEXT,
      model_a_success BOOLEAN,
      model_a_is_hate_speech BOOLEAN,
      model_a_error TEXT,
      model_b_name TEXT,
      model_b_success BOOLEAN,
      model_b_is_hate_speech BOOLEAN,
      model_b_error TEXT,
      comment TEXT
    )
  `);
}

async function insertResults(db: sqlite.Database, modelA: ModelResult, modelB: ModelResult): Promise<number> {
  let insertedCount = 0;
  
  // Extract comment from either model's output (they should have the same original data)
  const originalData = modelA.output || modelB.output || {};
  const comment = originalData.comment || originalData.text || null;
  const originalDataJson = JSON.stringify(originalData);
  
  // Insert the comparison result
  const stmt = db.prepare(`
    INSERT INTO hate_speech_results (
      original_data,
      model_a_name,
      model_a_success,
      model_a_is_hate_speech,
      model_a_error,
      model_b_name,
      model_b_success,
      model_b_is_hate_speech,
      model_b_error,
      comment
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  
  stmt.run([
    originalDataJson,
    modelA.model || "ModelA",
    modelA.success ? 1 : 0,
    modelA.is_hate_speech ? 1 : 0,
    modelA.error || null,
    modelB.model || "ModelB",
    modelB.success ? 1 : 0,
    modelB.is_hate_speech ? 1 : 0,
    modelB.error || null,
    comment
  ]);
  
  insertedCount++;
  stmt.finalize();
  
  return insertedCount;
}

export async function processJson(jsonInput: string): Promise<OutputResponse> {
  try {
    const data: DbInsertInput = JSON.parse(jsonInput);
    
    if (!data.ModelA || !data.ModelB) {
      return {
        success: false,
        error: "Missing ModelA or ModelB results in input data"
      };
    }

    // Initialize database connection
    const db = new sqlite.Database("hate_speech_results.db");
    
    try {
      // Initialize database schema
      await initializeDatabase(db);
      
      // Insert the results
      const insertedCount = await insertResults(db, data.ModelA, data.ModelB);
      
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
    Deno.exit(0);
  } catch (error) {
    standardOutput(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      inserted_records: 0
    }));
    Deno.exit(1);
  }
}

if (import.meta.main) {
  main();
}
