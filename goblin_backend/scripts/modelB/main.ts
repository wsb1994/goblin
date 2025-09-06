function standardOutput(output: string) {
  console.log(output);
}

export function processJson(jsonInput: string): object {
  try {
    const data = JSON.parse(jsonInput);
    const result = {
      success: true,
      output: data
    };
    return result;
  } catch (error: unknown) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

export function main() {
  const args = Deno.args;
  if (args.length === 0) {
    standardOutput(JSON.stringify({
      success: false,
      error: 'Please provide JSON input as argument',
      output: null
    }));
    Deno.exit(1);
  }

  const result = processJson(args[0]);
  standardOutput(JSON.stringify(result))
  Deno.exit(0);
}

if (import.meta.main) {
  main();
}
