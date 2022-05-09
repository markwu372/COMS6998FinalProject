# When and How CNN and GAN Generalize to Out-of-Distribution Category-Viewpoint Combinations

## Description of Project
Inspired by the paper “When and How CNNs Generalize to Out-Of-Distribution Category-Viewpoint Combinations”, in which the authors investigated when and how OOD generalization can be by evaluating CNNs trained to classify both object category and 3D viewpoint on OOD combinations, we explored and compared the generalization to Category-viewpoint Out-of-Distribution(OOD) data performances of different GAN models. 

When training a DNN, it is often assumed that the training and testing samples are drawn from the same distribution (in-distribution (InD) data). However, in practice, there likely exist abnormal test samples that are drawn from other distributions (out-of distribution (OOD) data), which may not belong to any of the classes that the model is trained on. We want to continue to explore the OOD generalization of CNN and GAN by following the implementation in the original paper. The three GAN models we used were SGAN, WGAN and ACGAN and three CNN models are ResNet, Inception V3, Xception. We ran the models on two datasets which were MNIST-Rotation and Biased-Cars datasets. At the end of the project, we visualized and compared the generalization performances between CNN and GAN models.

## Description of Repository

## Example commands to execute the code 

This fact may make the GANs more robust when dealing with the OOD problem. Run GAN_mnist_OOD/GAN_mnist.ipynb and GAN_biasedcar_OOD/GAN_biased_car.ipynb

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

### Biased-Cars Dataset
The Biased-Cars dataset is used to make comparison of the performance among three GANs since the Biased-Cars dataset are more comprehensive and challenged than the MNIST – Rotation dataset. Here is the result of different GAN models. For each GAN model, we also tried both shared and separate architectures.

![Alt text](/Diagrams/4.png?raw=true)
|:--:| 
| *Figure 3. Different GAN Performances on Biased-Cars Set(Separate and Shared)* |

We can observe that:
- For separate architecture:
    - On InD (seen) test set, when percentage of combinations is at 30% and 90 %, S-GAN has the best accuracy and when percentage of combinations is at 10% and 60 %, AC-GAN has the best accuracy.
    - On OOD (unseen) test set, when percentage of combinations is at 30%, 60% and 90 %, S-GAN has the best accuracy and when percentage of combinations is at 10%, AC-GAN has the best accuracy.

- For shared architecture:
    - On InD (seen) test set, when percentage of combination at 30%, 60% and 90%, S-GAN has the best accuracy and when percentage of combination at 10%, AC-GAN has the best accuracy.
    - On OOD (unseen) test set, S-GAN has the best accuracy for all data diversities

To sum up, when dealing with unseen OOD combinations, S-GAN has the best performance in the most data diversity range among all GANs. AC-GAN has the best performance when the data diversity and percentage of combination is low. W-GAN has the worst performance in overall.

We think that S-GAN performs the best among GAN models when dealing with out of distribution problem is becausethere are three types of data that S-GAN has access to: the first one is the labeled training data which the classifier needs to assign to correct classes; the second one is the unlabeled training data which the classifier only needs to be assigned to one of K classes; the third one is the fake data which is the images that have been generated by the generator and it should get assigned to the additional class K+1 (fake class). 
![Alt text](/Diagrams/3.png?raw=true)
|:--:| 
| *Figure 4. Architecture of S-GAN* |

Therefore, S-GAN can effectively improve the generalization ability of the discriminator by leveraging unlabeled information and the fake image created by the generator, which means both the category and viewpoint neurons can learn more information and become more robust when they see OOD combination in test set.




