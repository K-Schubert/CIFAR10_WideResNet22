# CIFAR10_WideResNet22
Wide ResNet 22 model trained on the CIFAR10 image dataset.  
- Data normalization, data augmentation (4px padding, random 32x32 cropping, random horizontal flipping (p=0.5)).
- Gradient clipping, Batch normalization, weight decay, learning rate scheduling, residual connections.
Model trained for 10 epochs on a GPU achieves 93% test accuracy.
