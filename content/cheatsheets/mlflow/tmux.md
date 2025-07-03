---
title: Tmux Cheat Sheet
date: 2025-07-03
draft: true
tags: ["tmux", "cheat sheet", "ml training", "data processing"]
categories: ["cheat sheets", "mlops", "data engineering"]
description: "A comprehensive guide to using tmux for managing long-running ML training jobs, data
---
# Tmux Cheat Sheet: Never Lose Your Training Jobs Again

Running ML training jobs, data processing pipelines, or any long-running process on remote servers comes with familiar pain points: SSH connections drop, you need to grab coffee during a 6-hour training run, or you simply want to start multiple experiments and monitor them independently. Enter tmux - the terminal multiplexer that keeps your processes alive and your sanity intact.

## Why Tmux for ML/Data Work?

- **Persistent sessions**: Training jobs survive SSH disconnections
- **Multiple processes**: Run different experiments simultaneously
- **Process monitoring**: Easy switching between training logs, system monitoring, and development
- **Collaboration**: Share sessions with team members for debugging
- **Resource isolation**: Organize different projects/experiments cleanly

## macOS vs Linux Considerations

**Using tmux on macOS:**
- Install via Homebrew: `brew install tmux`
- Copy/paste behavior differs: macOS clipboard integration requires additional config

**SSH from macOS to Linux servers:**
- Tmux runs on the **remote Linux machine**, not your local Mac
- Session persistence happens on the server, not your local machine
- Your Mac terminal is just a window into the remote tmux session
- Network interruptions won't affect remote tmux sessions

**Key insight**: When SSH'd into a Linux server, tmux behaves exactly like native Linux tmux because it's running on the Linux machine.

## Understanding Sessions vs Windows vs Panes

**Sessions**: Think of these as your entire workspace
- A collection of windows and panes
- Can have multiple sessions running simultaneously
- Each session can be detached and reattached
- Example: One session for training, another for monitoring, a third for development

**Windows**: Think of these as tabs in your browser
- Each window is a separate workspace
- Can have different processes running in each window
- Switch between windows like switching browser tabs
- Example: Window 1 for training, Window 2 for monitoring, Window 3 for development

**Panes**: Think of these as split screens within a single tab
- Divide one window into multiple sections
- All panes in a window are visible simultaneously
- Useful for side-by-side monitoring (logs + system stats)
- Example: Left pane shows training output, right pane shows GPU usage

## Essential Commands Reference

### Session Management

```bash
# Create new session (replace 'my-training' with your chosen name)
tmux new-session -s my-training

# Create session with specific name and start command
tmux new-session -s gpu-experiment -d 'python train_model.py'

# List all sessions
tmux list-sessions
tmux ls

# Attach to existing session (use your session name)
tmux attach-session -t my-training
tmux a -t my-training

# Detach from session (keeps processes running)
# Inside tmux: Ctrl+b then d

# Kill specific session (replace with your session name)
tmux kill-session -t my-training

# Kill all sessions
tmux kill-server
```

### Window Management

```bash
# Create new window
# Inside tmux: Ctrl+b then c

# Rename current window
# Inside tmux: Ctrl+b then ,

# List windows
# Inside tmux: Ctrl+b then w

# Switch to window by number
# Inside tmux: Ctrl+b then 0-9

# Switch to next/previous window
# Inside tmux: Ctrl+b then n/p

# Kill current window
# Inside tmux: Ctrl+b then &
```

### Pane Management

```bash
# Split window vertically (side by side)
# Inside tmux: Ctrl+b then %

# Split window horizontally (top/bottom)
# Inside tmux: Ctrl+b then "

# Navigate between panes
# Inside tmux: Ctrl+b then arrow keys

# Resize pane
# Inside tmux: Ctrl+b then Ctrl+arrow keys

# Toggle pane zoom (full screen)
# Inside tmux: Ctrl+b then z

# Kill current pane
# Inside tmux: Ctrl+b then x
```

### Copy Mode & Scrolling

```bash
# Enter copy mode (for scrolling/copying)
# Inside tmux: Ctrl+b then [

# In copy mode:
# - Use arrow keys or vi keys (j/k/h/l) to navigate
# - Space to start selection
# - Enter to copy selection
# - q to exit copy mode

# Paste copied content
# Inside tmux: Ctrl+b then ]

# Show paste buffer
# Inside tmux: Ctrl+b then =
```

## Logging and Output Management

**Important**: Tmux itself doesn't automatically log your output - you need to set this up explicitly.

### Manual Logging Setup

```bash
# Redirect all output to a log file
python train_model.py > training.log 2>&1

# Redirect output and still see it in terminal
python train_model.py 2>&1 | tee training.log

# Include timestamps in logs
python train_model.py 2>&1 | while read line; do echo "$(date): $line"; done | tee training.log
```

### Tmux Built-in Logging

```bash
# Start logging current pane (creates tmux-output.log)
# Inside tmux: Ctrl+b then :
# Then type: pipe-pane -o 'cat >> ~/tmux-output.log'

# Stop logging
# Inside tmux: Ctrl+b then :
# Then type: pipe-pane

# Log with custom filename
# Inside tmux: Ctrl+b then :
# Then type: pipe-pane -o 'cat >> ~/my-experiment-$(date +%Y%m%d).log'
```

### Session History and Scrollback

```bash
# Save entire pane history to file
# Inside tmux: Ctrl+b then :
# Then type: capture-pane -p > ~/session-history.txt

# Save with more lines of history
# Inside tmux: Ctrl+b then :
# Then type: capture-pane -p -S -3000 > ~/full-history.txt

# Configure scrollback buffer size (add to ~/.tmux.conf)
set-option -g history-limit 10000
```

### macOS Clipboard Integration

```bash
# Add to ~/.tmux.conf for macOS clipboard integration
# Install reattach-to-user-namespace first: brew install reattach-to-user-namespace
set-option -g default-command "reattach-to-user-namespace -l bash"

# Copy selection to macOS clipboard
bind-key -T copy-mode-vi 'y' send-keys -X copy-pipe-and-cancel 'reattach-to-user-namespace pbcopy'

# Or for newer tmux versions
bind-key -T copy-mode-vi 'y' send-keys -X copy-pipe-and-cancel 'pbcopy'
```

## ML-Specific Workflows

### Training Job Setup

```bash
# Start training session with descriptive name (choose your own name)
tmux new-session -s bert-finetuning

# Create monitoring layout
# Window 0: Training logs
# Window 1: System monitoring
# Window 2: Development/debugging

# In Window 0 - start training with logging
python train_bert.py --epochs 50 --batch-size 32 2>&1 | tee training_$(date +%Y%m%d_%H%M).log

# Detach and create monitoring window
# Ctrl+b, d, then:
tmux new-window -t bert-finetuning -n monitoring
htop

# Create development window
tmux new-window -t bert-finetuning -n dev
```

### Multi-GPU Training

```bash
# Create session for distributed training
tmux new-session -s distributed-training

# Split into panes for different GPUs
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Now you have 4 panes for monitoring 4 GPUs
# Pane 0: CUDA_VISIBLE_DEVICES=0 python train.py
# Pane 1: CUDA_VISIBLE_DEVICES=1 python train.py
# Pane 2: nvidia-smi -l 1
# Pane 3: tail -f training.log
```

### Experiment Management

```bash
# Create session for experiment tracking
tmux new-session -s experiments

# Window layout:
# experiments:0 - experiment-1
# experiments:1 - experiment-2
# experiments:2 - monitoring

# Start multiple experiments
tmux new-window -t experiments -n exp-1
python train.py --lr 0.001 --model resnet50

tmux new-window -t experiments -n exp-2
python train.py --lr 0.01 --model resnet18

tmux new-window -t experiments -n monitoring
watch -n 5 'nvidia-smi && echo "=== GPU Usage ===" && df -h'
```

## Advanced Configuration

### Custom Key Bindings

Create `~/.tmux.conf`:

```bash
# Change prefix key (default Ctrl+b)
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Enable mouse support
set -g mouse on

# Improve colors
set -g default-terminal "screen-256color"

# Start windows and panes at 1
set -g base-index 1
setw -g pane-base-index 1

# Vi mode for copy
setw -g mode-keys vi

# Reload config
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# Better splits
bind | split-window -h
bind - split-window -v

# Increase scrollback buffer
set-option -g history-limit 10000

# macOS specific - clipboard integration
# First install: brew install reattach-to-user-namespace
set-option -g default-command "reattach-to-user-namespace -l $SHELL"
bind-key -T copy-mode-vi 'y' send-keys -X copy-pipe-and-cancel 'pbcopy'
```

### Status Bar Customization

```bash
# Add to ~/.tmux.conf
set -g status-bg black
set -g status-fg white
set -g status-left '#[fg=green]#S #[fg=white]| '
set -g status-right '#[fg=yellow]#(uptime | cut -d "," -f 1) #[fg=white]| #[fg=cyan]%Y-%m-%d %H:%M'
set -g status-left-length 20
set -g status-right-length 50
```

## Quick Reference Card

| Action | Command |
|--------|---------|
| New session | `tmux new -s name` |
| Attach session | `tmux a -t name` |
| List sessions | `tmux ls` |
| Detach | `Ctrl+b d` |
| New window | `Ctrl+b c` |
| Split vertical | `Ctrl+b %` |
| Split horizontal | `Ctrl+b "` |
| Navigate panes | `Ctrl+b arrow` |
| Zoom pane | `Ctrl+b z` |
| Copy mode | `Ctrl+b [` |
| Paste | `Ctrl+b ]` |
| Kill session | `tmux kill-session -t name` |

## Pro Tips

1. **Name your sessions descriptively**: Use meaningful names like `bert-training`, `data-preprocessing`, `model-evaluation` (you choose the names)
2. **Use detach liberally**: Start job, detach, go grab coffee, reattach to check progress
3. **Keep a monitoring window**: Always have htop/nvidia-smi running in a separate window
4. **Log everything**: Always redirect training output to files: `python train.py 2>&1 | tee experiment.log`
5. **Use tmux with nohup**: For extra safety, combine with `nohup` for critical jobs
6. **Session templates**: Create shell scripts to set up common tmux layouts
7. **macOS users**: Install iTerm2 for better tmux integration than Terminal.app
8. **Remote work**: Remember tmux runs on the server, not your local Mac

## Common Pitfalls

- **Don't forget to detach**: Closing terminal without detaching kills the session
- **Memory management**: Long-running sessions can accumulate memory leaks
- **Log rotation**: Training logs can fill up disk space quickly - use logrotate or manual cleanup
- **GPU memory**: Detached sessions hold GPU memory - clean up unused sessions
- **macOS clipboard**: Without proper config, copy/paste between tmux and macOS clipboard won't work
- **No automatic logging**: Tmux doesn't log output by default - you must set it up manually

---

*Happy training! May your models converge and your gradients flow smoothly.*