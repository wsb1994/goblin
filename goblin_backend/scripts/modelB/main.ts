

export function processJson(jsonInput: string): object {
  try {
    const data = JSON.parse(jsonInput);
    const result = {
      ...data,
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
    return {
      success: false,
      error: 'Please provide JSON input as argument'
    };
  }

  const result = processJson(args[0]);
  return {
    success: true,
    result
  }
}

// Learn more at https://docs.deno.com/runtime/manual/examples/module_metadata#concepts
if (import.meta.main) {
  main();
}
