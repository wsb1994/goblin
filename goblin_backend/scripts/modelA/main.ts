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
*/

function standardOutput(output: string) {
  console.log(output);
}

interface InputData {
  id?: string;
  text?: string;
  label?: number;
  comment?: string;
  [key: string]: any;
}

interface OutputResponse {
  success: boolean;
  is_hate_speech?: boolean;
  model: string;
  output?: any;
  error?: string;
}

const MODEL_NAME = "claude-sonnet-4-20250514";

async function checkHateSpeech(comment: string): Promise<boolean> {
  const apiKey = Deno.env.get("ANTHROPIC_API_KEY");
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY environment variable is required");
  }

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        max_tokens: 100,
        messages: [
          {
            role: "user",
            content: `You are a hate speech detection system. Analyze the given text and determine if it contains hate speech. Hate speech includes content that attacks, threatens, or incites violence against individuals or groups based on attributes like race, religion, ethnic origin, sexual orientation, disability, or gender. Respond with only 'YES' if it contains hate speech, or 'NO' if it does not. Utilize EU definitions for hate speech by default as they should align with global standards strongly.

Text to analyze: ${comment}`
          }
        ]
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const result = data.content?.[0]?.text?.trim().toUpperCase();
    return result === "YES";
  } catch (error) {
    console.error("Error calling Claude API:", error);
    throw new Error("Failed to analyze comment for hate speech");
  }
}

export async function processJson(jsonInput: string): Promise<OutputResponse> {
  try {
    const data: InputData = JSON.parse(jsonInput);
    
    // Handle both old format (comment) and new format (text)
    const textToAnalyze = data.text || data.comment;
    
    if (!textToAnalyze) {
      return {
        success: false,
        model: MODEL_NAME,
        error: "Missing 'text' or 'comment' field in input data",
        output: data
      };
    }

    const isHateSpeech = await checkHateSpeech(textToAnalyze);
    
    return {
      success: true,
      model: MODEL_NAME,
      is_hate_speech: isHateSpeech,
      output: data
    };
  } catch (error: unknown) {
    return {
      success: false,
      model: MODEL_NAME,
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
      output: null
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
      output: null
    }));
    Deno.exit(1);
  }
}

if (import.meta.main) {
  main();
}
