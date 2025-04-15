#!/bin/bash
LOG=logs/app_$(date +%Y%m%d_%H%M%S).log;
mkdir -p logs;
python ./web.py --log "$LOG" 2>&1 | tee "$LOG";