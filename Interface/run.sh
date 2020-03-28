


for model in 'alexnet' 'densenet121' 'densenet161'  'densenet169' 'densenet201' 'squeezenet1_0' 'squeezenet1_1'  'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' 'wide_resnet101_2' 'wide_resnet50_2'  'vgg11'  'resnet50' 'resnet101' 'resnet152' 'resnet18' 'resnet34';
do
	python explore_conv_only.py -m $model >> conv_only_dse_res &
done

wait
