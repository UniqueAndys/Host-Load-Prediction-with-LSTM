output_dim_arr=(6 12 18 24 30 36)
grid_arr=("axp0" "axp7" "sahara" "themis")
for output_dim in ${output_dim_arr[@]}; do
	echo "The output_dim is $output_dim"
	for grid in ${grid_arr[@]}; do
		echo "The grid is $grid"
		python esn_mse_main.py --output_dim=$output_dim --grid=$grid >> 0902.txt
	done
done
