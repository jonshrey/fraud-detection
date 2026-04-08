FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port for Gradio/HF Spaces
EXPOSE 7860

# Run the space app
CMD ["python", "space_app.py"]