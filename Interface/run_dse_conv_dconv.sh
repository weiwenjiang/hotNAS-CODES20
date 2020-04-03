



for model in 'mnasnet0_5' 'mobilenet_v2' 'mnasnet1_0' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 'proxyless_gpu' 'proxyless_cpu' 'proxyless_mobile';
do
#	touch res/$model
	python explore_conv_dconv.py -m $model > res/$model.res &
done

wait
