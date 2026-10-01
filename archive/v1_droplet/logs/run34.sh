cd /home/claude/droplet
nohup python3 -m src.phases 3 > logs/phase3.log 2>&1 &
nohup python3 -m src.phases 4 > logs/phase4.log 2>&1 &
