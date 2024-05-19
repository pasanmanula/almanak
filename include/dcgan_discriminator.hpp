#ifndef ALMANAK_DCGAN_DISCRIMINATOR_HPP_
#define ALMANAK_DCGAN_DISCRIMINATOR_HPP_

#include <torch/torch.h>

/*
 * This network defines the discriminator module, which consists of a series of
 * 2D convolutions, batch normalizations and LeakyReLU activation units and finally a sigmoid neuron.
 */
class DCGANDiscriminatorImpl : public torch::nn::Module {
public:
  /*
  * Construct the discriminator architecture.
  * 
  */
  explicit DCGANDiscriminatorImpl();  
  /*
  * Defines the forward computation of the discriminator network.
  * 
  */
  torch::Tensor forward(torch::Tensor input);  

private:
  torch::nn::Conv2d conv1_, conv2_, conv3_, conv4_;
  torch::nn::BatchNorm2d batch_norm1_, batch_norm2_;
};

#endif  // ALMANAK_DCGAN_DISCRIMINATOR_HPP_

TORCH_MODULE(DCGANDiscriminator);