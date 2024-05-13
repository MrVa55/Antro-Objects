#!/bin/bash
nc -l -p 5500| aplay 2>/dev/null &
aplay_pid=$(ps aux |grep [^]]aplay|awk '{print $2}')
echo "aplay_pid: $aplay_pid"
ssh philipschubell@34.118.67.82 -R 5500:localhost:5500 -f  "touch character.file"; echo 'connected..'
ssh_pid=$(ps aux |grep ssh|grep 5500|awk '{print $2}')
trap 'ssh philipschubell@34.118.67.82 "rm -f character.file"; kill -9 $ssh_pid 2> /dev/null ; kill -9 $aplay_pid 2> /dev/null ; exit 1' EXIT HUP INT TERM
echo "ssh_pid: $ssh_pid"

while true; do
  sleep 1
done
