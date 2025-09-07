FROM python:3.13-slim

WORKDIR /app
COPY . /app/src

# Install system dependencies + Deno
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && curl -fsSL https://deno.land/install.sh | sh \
    && mv /root/.deno/bin/deno /usr/local/bin/deno \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for dependency install
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install --no-cache-dir uv \
    && which uv && uv --version

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

EXPOSE 8001

# Run the app
CMD ["uv", "run", "main.py"]
