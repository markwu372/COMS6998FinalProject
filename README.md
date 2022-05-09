# When and How CNN and GAN Generalize to Out-of-Distribution Category-Viewpoint Combinations

## Description of Project
From the paper “When and How CNNs Generalize to Out-Of-Distribution Category-Viewpoint Combinations”, the authors investigated when and how OOD generalization can be by evaluating CNNs trained to classify both object category and 3D viewpoint on OOD combinations, and identified the neural mechanisms that facilitate such OOD generalization. Recent works suggest that CNNs can hardly to generalize OOD category-viewpoint combinations


In the first part of the experiments, different CNNs architectures and different category-viewpoint architectures are tried to explore their performance of Out-Of-Distribution Category-Viewpoint Generalization on MNIST – Rotation dataset (MNIST-Position and MNIST-Scale dataset). MNIST-Position was created by placing MNIST images into one of nine possible locations in an empty 3-by-3 grid. Images are resized to one of nine possible sizes followed by zero-padding. Images of the digit 9 were left out in both these datasets, ensuring nine categories and nine viewpoints classes (total of 81 category- viewpoint combinations). Total 5 different kinds of CNN architectures are tried: Resnet 18, Resnet 34, Resnet 50, inception V3, Xception. For those CNN architectures, we also tried ‘Separate’, ‘Shared’, and different ‘Split’ architectures for category and viewpoint. Run CNN_mnist.ipynb


In the second part of the experiments, different Generative Adversarial Network (GAN) architectures and different category-viewpoint architectures are tried to explore their performance of Out-Of-Distribution Category Viewpoint Generalization on MNIST - Rotation datasets and Biased-Cars dataset. The inspiration and motivation of trying GANs on OOD problem is that, during the training process of GANs networks, the generator may generate images which category and viewpoint combination are out of the distribution of the trainset, and then the discriminator will then learn from these out of distribution category and viewpoint combinations from the images generated by the generator. This fact may make the GANs more robust when dealing with the OOD problem. Run GAN_mnist_OOD/GAN_mnist.ipynb and GAN_biasedcar_OOD/GAN_biased_car.ipynb


## Description of Repository

## Example commands to execute the code 

## Results (including charts/tables) and observations  
### Evaluation Criteria
We will use the geometric mean of category and viewpoint classification accuracy as mentioned in the original paper of “When and How CNNs Generalize to Out-Of-Distribution Category-Viewpoint Combinations” to evaluate the OOD generalization of different CNN and GAN architectures.

### MNIST-Rotation Dataset
In the first part of our experiments, different CNNs architectures are tried to explore their performance of Out-Of-Distribution Category-Viewpoint Generalization on MNIST–Rotation dataset (MNIST- Position and MNIST-Scale dataset). MNIST-Rotation was created by placing MNIST images into one of nine possible locations in an empty 3-by-3 grid. Images are resized to one of nine possible sizes followed by zero-padding. Images of the digit 9 were left out in both these datasets, ensuring nine categories and nine viewpoint classes (total of 81 category- viewpoint combinations). We tried ‘Separate’ and ‘Shared’ architectures for category and viewpoint. 
![Alt text](/Diagrams/2.png?raw=true)
|:--:| 
| *Figure 1. Different CNN Performances on MNIST-Rotation Set(Separate and Shared)* |


We can see Xception and Inception V3 has the relatively better performances among all the shared and separate architectures. The reason might be due to the Depthwise Separable Convolution. Compared with traditional spatial wise convolution, Depth-wise Separable Convolution deals not just with the spatial dimensions, but with the depth dimension — the number of channels — as well. An input image may have 3 channels: RGB. After a few convolutions, an image may have multiple channels. We can imagine each channel as a particular interpretation of that image.In some channels, the category information is stored, while others channels store the viewpoint information. In some channels both category and viewpoint information are stored. By doing Depth-wise separable convolution, we can train category neurons and viewpoint neurons separately, which will not disturb each other. Therefore we can regard Depth-wise Separable Convolution as a variant of ‘Separate’ architecture because they have similar effects, so that the Depth-wise Separable Convolution in Xception and inception V3 can help to improve the performance of predicting OOD combinations.

![Alt text](/Diagrams/1.png?raw=true)
|:--:| 
| *Figure 2. Different GAN Performances on MNIST-Rotation Set(Separate and Shared)* |

We can observe that: 
- S-GAN has the best performances among GANs while Xception has the best performance among CNNs
- Among the performance on InD combinations test sets, Xception has a better performance than S-GAN in both ‘separate’ and ‘shared’ architectures.
- Among the performance on OOD combinations in test sets, Xception also has better performance than S-GAN in ‘separate’ architecture, but In the ‘shared’ architecture, S-GAN has a much better performance than Xception.
We think this may be caused by the fact that, when predicating category-viewpoint on out of distribution data in a separate network, the neurons only focus on either category task or viewpoint task. Since Xception has a more complicated network, it will perform better than SGAN. 
However, when predicating category-viewpoint on out of distribution data in a shared network, neurons have to focus on both category and viewpoint tasks. Because of the adversarial property during training, the generator may generate images in which category and viewpoint combinations are out of the distribution of the training set, and then the category and viewpoint neurons in the discriminator will then learn from these out of distribution combinations from the generated images. Therefore, S-GAN can perform better than Xception when dealing with out of distribution problems in a shared network.


