#!/bin/bash

# Variables - replace these with your info
KEY_PATH="$HOME/.ssh/aws-ec2-keys.pem"
USER="ubuntu"
EC2_IP="52.64.161.84"
REMOTE_DIR="/home/$USER/bert_training"

# Files to copy
FILES="finetune_bert.py magnus_finetune_for_bert.jsonl move_vocab.json"

echo "Creating remote directory..."
ssh -i $KEY_PATH $USER@$EC2_IP "mkdir -p $REMOTE_DIR"

echo "Copying files to remote..."
scp -i $KEY_PATH $FILES $USER@$EC2_IP:$REMOTE_DIR

echo "Installing dependencies on remote..."
ssh -i $KEY_PATH $USER@$EC2_IP << EOF
    sudo apt update
    sudo apt install -y python3-pip
    pip3 install --upgrade pip
    pip3 install torch transformers chess
EOF

echo "Running training script on remote..."
ssh -i $KEY_PATH $USER@$EC2_IP << EOF
    cd $REMOTE_DIR
    python3 finetune_bert.py --data_path magnus_finetune_for_bert.jsonl --move_vocab_path move_vocab.json --output_dir ./bert_chess_model --epochs 3 --batch_size 16 --lr 5e-5
EOF

echo "Done!"
