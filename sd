#!/bin/bash

REMOTE_USER="$1"

DATA_DIR="$(pwd)/data"
LOCAL_DIR="$(pwd)"

REMOTE_HOST="edward.inf-ra.uni-jena.de"
REMOTE_DIR="/home/${REMOTE_USER}/data"

PORT=2025

# copy from local to remote
echo "Copying local data from $DATA_DIR to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR..."
scp -P $PORT -r "$DATA_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
if [ $? -ne 0 ]; then
    echo "Error occurred while copying data to remote."
    exit 1
fi

# connect to the remote host
echo "Connecting to remote server..."
ssh -p $PORT "$REMOTE_USER@$REMOTE_HOST"
if [ $? -ne 0 ]; then
    echo "Error occurred while connecting to the remote server."
    exit 1
fi

# remove the local data directory
echo "Remove local data directory..."
rm -r "$DATA_DIR"
if [ $? -ne 0 ]; then
    echo "Error occurred while copying data from remote."
    exit 1
fi

# copy from remote to local
echo "Copying data from $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR to $LOCAL_DIR..."
scp -P $PORT -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"
if [ $? -ne 0 ]; then
    echo "Error occurred while copying data from remote."
    exit 1
fi

# remove the data directory from the remote machine after copying
echo "Removing data folder from remote machine..."
ssh -p $PORT "$REMOTE_USER@$REMOTE_HOST" "rm -r $REMOTE_DIR"
if [ $? -ne 0 ]; then
    echo "Error occurred while removing the data folder from the remote machine."
    exit 1
fi