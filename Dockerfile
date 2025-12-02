# Use Python 3.10 (TensorFlow compatible)
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose port (not used by Telegram Bot, but Render needs it)
EXPOSE 10000

# Start the bot
CMD ["python", "bot.py"]
