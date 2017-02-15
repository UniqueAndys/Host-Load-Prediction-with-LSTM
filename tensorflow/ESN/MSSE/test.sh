#!/bin/bash

interval_arr=(1 2 3 4 5 6 7 8)
for interval in ${interval_arr[@]}; do
	echo "The interval is $interval"
	python esn_msse_main.py --interval=$interval >> 1013.txt
done
