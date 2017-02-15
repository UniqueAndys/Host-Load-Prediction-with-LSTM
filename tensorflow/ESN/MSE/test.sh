#!/bin/bash
if [ ]; then
	output_dim_arr=(36 30 24 18 12 6)
	for output_dim in ${output_dim_arr[@]}; do
		echo "The output_dim is $output_dim"
		start=$(date "+%M")
		python test.py --output_dim=$output_dim --batch_size=128 >> 36_3.txt
		python test.py --output_dim=$output_dim --batch_size=64 >> 36_3.txt
		now=$(date "+%M")
		time=$(($now-$start))
		echo "time used: $time minites"
	done

output_dim_arr=(36 30 24 18 12 6)
aug_arr=(1 3 6 12)
for output_dim in ${output_dim_arr[@]}; do
	echo "The output_dim is $output_dim"
	for aug in ${aug_arr[@]}; do
		echo "The aug is $aug"
		python test_new.py --aug=$aug --output_dim=$output_dim >> ies_11.txt
	done
done

output_dim_arr=(6 12 18 24 30 36)
for output_dim in ${output_dim_arr[@]}; do
	echo "The output_dim is $output_dim"
	python esn_mse_save_logits.py --output_dim=$output_dim >> 0722_2.txt
done
fi

output_dim_arr=(6 12 18 24 30 36)
for output_dim in ${output_dim_arr[@]}; do
	echo "The output_dim is $output_dim"
	python esn_mse_main.py --output_dim=$output_dim >> 1013.txt
done
