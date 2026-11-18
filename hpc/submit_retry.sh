#!/bin/bash
# Submit queued work as the per-user queue limit allows: each line of hpc/packs/queue.txt is qsub arguments;
# retries every 10 min until each is accepted (nohup on the login node). Log: logs/submit_retry.log
cd ~/cricket-project || exit 1
while read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    until id=$(qsub $line 2>/dev/null); do sleep 600; done
    echo "$(date '+%F %T') $id  <- $line"
done < hpc/packs/queue.txt
echo "$(date '+%F %T') all submitted"
