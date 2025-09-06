#!/bin/bash

# Execute Deno script with JSON argument and store the output
output=$(deno run --allow-read main.ts '{"test": "example"}')

# Check if the output contains "success": true
if echo "$output" | grep -q '"success":true'; then
  echo "true"
else
  echo "false"
fi

exit 0