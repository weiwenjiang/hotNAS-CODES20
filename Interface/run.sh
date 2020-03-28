


#for model in 'alexnet' 'densenet121' 'densenet161'  'densenet169' 'densenet201' 'squeezenet1_0' 'squeezenet1_1'  'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' 'wide_resnet101_2' 'wide_resnet50_2'  'vgg11'  'resnet50' 'resnet101' 'resnet152' 'resnet18' 'resnet34';
for model in 'mnasnet0_5' 'mobilenet_v2' 'mnasnet1_0' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0';
do
	python explore_conv_dconv.py -m $model >> conv_dconv_dse_res &
done

wait
