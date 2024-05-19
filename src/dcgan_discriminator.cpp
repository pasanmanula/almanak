#include "dcgan_discriminator.hpp"

DCGANDiscriminatorImpl::DCGANDiscriminatorImpl()
    : conv1_(register_module(
             "conv1",
             torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4)
             .stride(2).padding(1).bias(false)))),
      conv2_(register_module(
             "conv2",
             torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4)
             .stride(2).padding(1).bias(false)))),
      conv3_(register_module(
             "conv3",
             torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4)
             .stride(2).padding(1).bias(false)))),
      conv4_(register_module(
             "conv4",
             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3)
             .stride(1).padding(0).bias(false)))),
      batch_norm1_(register_module(
             "batch_norm1",
             torch::nn::BatchNorm2d(128))),
      batch_norm2_(register_module(
             "batch_norm2",
             torch::nn::BatchNorm2d(256)))
{ }

torch::Tensor DCGANDiscriminatorImpl::forward(torch::Tensor x)
{
    x = torch::leaky_relu(conv1_->forward(x), 0.2);
    x = torch::leaky_relu(batch_norm1_->forward(conv2_->forward(x)), 0.2);
    x = torch::leaky_relu(batch_norm2_->forward(conv3_->forward(x)), 0.2);
    x = torch::sigmoid(conv4_->forward(x));
  return x;
}

