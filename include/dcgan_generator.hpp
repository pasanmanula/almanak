#ifndef ALMANAK_DCGAN_GENERATOR_HPP_
#define ALMANAK_DCGAN_GENERATOR_HPP_

#include <torch/torch.h>

/*
 * This network defines the generator module, which consists of a series of
 * transposed 2D convolutions, batch normalizations and ReLU activation units.
 */
class DCGANGeneratorImpl : public torch::nn::Module {
public:
  /*
  * Construct the generator architecture.
  * 
  */
  explicit DCGANGeneratorImpl(int noise_size);  
  /*
  * Defines the forward computation of the generator network.
  * 
  */
  torch::Tensor forward(torch::Tensor input);  

private:
  torch::nn::ConvTranspose2d conv1_, conv2_, conv3_, conv4_;
  torch::nn::BatchNorm2d batch_norm1_, batch_norm2_, batch_norm3_;
};

#endif  // ALMANAK_DCGAN_GENERATOR_HPP_

TORCH_MODULE(DCGANGenerator);