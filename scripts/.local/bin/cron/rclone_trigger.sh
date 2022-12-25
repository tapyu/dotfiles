#!/bin/bash

# Unix shell script to bisynch iff a event has been done after 5min of the last modification 
# $1 -> drive path
# $2 -> local path
# $3 -> remote path

CURRENT_STAT=$(stat --format='%z' /home/tapyu/books | awk '{ print $1,$2 }')
if [[ -f /home/tapyu/.cache/rclone/last_sync.log ]]; then
    if [[ $CURRENT_STAT != $(cat /home/tapyu/.cache/rclone/last_sync.log) ]]; then
        rclone bisync $2 $3
        LAST_SYNC=$DATE
    fi
else
    rclone bisync $2 $3
    echo $CURRENT_STAT > /home/tapyu/.cache/rclone/last_sync.log
fi