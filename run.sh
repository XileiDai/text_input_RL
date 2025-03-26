learning_rate="0.000001 0.00001 0.00005 0.000005"
batch_size="16 8"
total_epoch='50'
max_buffer_size='2000 500 1000'

for batch_size_i in $batch_size
do
	for total_epoch_i in $total_epoch
	do
			for max_buffer_size_i in $max_buffer_size
			do
				for learning_rate_i in $learning_rate
				do
					python test_dqn.py --gradient_steps 5 --epsilon_start 0.2 --learning_rate $learning_rate_i --max_buffer_size $max_buffer_size_i --total_epoch $total_epoch_i --batch_size $batch_size_i
 				done
			done
	done
done