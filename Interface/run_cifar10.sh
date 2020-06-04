


#for model in 'densenet121' 'densenet169' 'resnet18' 'resnet34' 'resnet50' ;
##for model in 'mnasnet0_5' 'mobilenet_v2' 'mnasnet1_0' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 'proxyless_cpu' 'proxyless_gpu' 'proxyless_mobile' 'FBNET';
#do
##	touch res/$model
#	python explore_conv_only.py -m $model > res/$model.res &
##	python explore_conv_dconv.py -m $model > res/$model.res &
#done


for model in 'mobilenet_v2' ;
do
  python explore_conv_dconv.py -m $model > res/$model.res &
done

wait
