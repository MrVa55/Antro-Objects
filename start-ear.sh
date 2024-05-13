#!/bin/bash
ssh philipschubell@34.118.67.82 -L 4500:localhost:4500 -f -o ExitOnForwardFailure=yes "touch character.file"; echo 'connected..'
ssh_pid=$(lsof -t -i @localhost:4500 -sTCP:listen)
trap 'ssh philipschubell@34.118.67.82 "rm -f character.file"; kill -9 $ssh_pid 2> /dev/null ; exit 1' EXIT HUP INT TERM
arecord -f S16_LE -c1 -r 16000 -t raw -D default|nc localhost 4500
