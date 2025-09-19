#!/bin/bash

echo "Checking the currently running 'ollama serve' processes..."
OLLAMA_PROCESSES=$(lsof -iTCP -sTCP:LISTEN | grep 'ollama')

if [ -z "$OLLAMA_PROCESSES" ]; then
    echo "No 'ollama serve' processes are currently running."
else
    echo "List of running 'ollama serve' processes:"
    echo "$OLLAMA_PROCESSES"
    echo ""
fi

read -p "Enter the port to start or stop: " OLLAMA_HOST_PORTS

if [ -z "$OLLAMA_HOST_PORTS" ]; then    
    echo "You must enter a port number. Exiting the script."
    exit 1
fi

for OLLAMA_HOST_PORT in $OLLAMA_HOST_PORTS; do
    if lsof -iTCP -sTCP:LISTEN | grep -q $OLLAMA_HOST_PORT; then
        PID=$(lsof -ti :$OLLAMA_HOST_PORT)
        echo "Terminating the 'ollama serve' process running on port $OLLAMA_HOST_PORT (PID: $PID)..."
        kill -9 $PID
        echo "Process has been terminated."
    else
        read -p "Enter the model to run on port $OLLAMA_HOST_PORT: " MODEL_NAME

        if [ -z "$MODEL_NAME" ]; then
            echo "You must specify a model. Exiting the script."
            exit 1
        fi

        read -p "Enter the gpu to run (e.g., 0 for single, or 0,1 for multi): " GPU

        if [ -z "$GPU" ]; then
            echo "You must specify a gpu. Exiting the script."
            exit 1
        fi

        echo "Starting 'ollama serve' on port $OLLAMA_HOST_PORT."
        CUDA_VISIBLE_DEVICES=$GPU OLLAMA_HOST="localhost:$OLLAMA_HOST_PORT" ollama serve &
        sleep 1
        echo "Executing installation command."
        curl -X POST http://localhost:$OLLAMA_HOST_PORT/api/pull -d "{\"model\":\"$MODEL_NAME\"}"
    fi
done



