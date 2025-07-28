# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user
RUN useradd -m myuser

# Copy the requirements file and install dependencies
COPY --chown=myuser:myuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=myuser:myuser . .

# Add the project root to the Python path
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Switch to the non-root user
USER myuser

# Set the default command to run when the container starts
CMD ["/bin/bash"]
