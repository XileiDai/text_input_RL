learning_rate="0.000001"
batch_size="256"
total_epoch='200'
max_buffer_size='10000'

for batch_size_i in $batch_size
do
	for total_epoch_i in $total_epoch
	do
			for max_buffer_size_i in $max_buffer_size
			do
				for learning_rate_i in $learning_rate
				do
					python test_dqn.py --gradient_steps 5 --learning_rate $learning_rate_i --max_buffer_size $max_buffer_size_i --total_epoch $total_epoch_i --batch_size $batch_size_i
 				done
			done
	done
done