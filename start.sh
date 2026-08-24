#!/bin/bash
# ==========================================
# Script: start.sh
# Purpose: Launch a 4-process or 5-process training cluster in separate tmux windows
# ==========================================

# Shared environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Define the virtual environment activation command
VENV_ACTIVATE="source .venv/bin/activate"

echo "Please select a startup mode:"
echo "[1] 4-process cluster (Server + Middle1-2 + Client)"
echo "[2] 5-process cluster (Server + Middle1-3 + Client)"
read -p "Enter 1 or 2: " choice

# Use a consistent tmux session name
SESSION_NAME="train_cluster"

# Close any existing session with the same name to prevent conflicts
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create a new detached tmux session
tmux new-session -d -s $SESSION_NAME

case $choice in
    1)
        echo "[Starting] 4-process cluster mode..."
        
        # Window 0: Start Server (Rank 0)
        tmux send-keys -t $SESSION_NAME:0 "$VENV_ACTIVATE && export RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 && python scripts/train_pretrain_france_server1.py" C-m
        
        # Window 1: Start Middle 1 (Rank 1)
        sleep 10
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:1 "$VENV_ACTIVATE && export RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 && python scripts/train_pretrain_france_middle_server1.py" C-m
        
        # Window 2: Start Middle 2 (Rank 2)
        sleep 20
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:2 "$VENV_ACTIVATE && export RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 && python scripts/train_pretrain_france_middle_server2.py" C-m
        
        # Window 3: Start Client (Rank 3)
        sleep 30
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:3 "$VENV_ACTIVATE && export RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 && python scripts/train_pretrain_france_client1.py" C-m
        ;;
        
    2)
        echo "[Starting] 5-process cluster mode..."
        
        # Window 0: Start Server (Rank 0)
        tmux send-keys -t $SESSION_NAME:0 "$VENV_ACTIVATE && export RANK=0 LOCAL_RANK=0 WORLD_SIZE=5 && python scripts/train_pretrain_france_server.py" C-m
        
        # Window 1: Start Middle 1 (Rank 1)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:1 "$VENV_ACTIVATE && export RANK=1 LOCAL_RANK=1 WORLD_SIZE=5 && python scripts/train_pretrain_france_middle_server1.py" C-m
        
        # Window 2: Start Middle 2 (Rank 2)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:2 "$VENV_ACTIVATE && export RANK=2 LOCAL_RANK=2 WORLD_SIZE=5 && python scripts/train_pretrain_france_middle_server2.py" C-m
        
        # Window 3: Start Middle 3 (Rank 3)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:3 "$VENV_ACTIVATE && export RANK=3 LOCAL_RANK=3 WORLD_SIZE=5 && python scripts/train_pretrain_france_middle_server3.py" C-m
 
        # Window 4: Start Client 4 (Rank 4)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:4 "$VENV_ACTIVATE && export RANK=4 LOCAL_RANK=4 WORLD_SIZE=5 && python scripts/train_pretrain_france_client.py" C-m
        ;;
        
    *)
        echo "Invalid selection. Exiting."
        exit 1
        ;;
esac

echo "[Completed] All processes have been started in separate tmux windows."
echo "Tip: Run 'tmux attach -t $SESSION_NAME' to view the session. Press Ctrl+B, then D to detach."
