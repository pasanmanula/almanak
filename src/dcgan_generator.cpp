#include "dcgan_generator.hpp"

DCGANGeneratorImpl::DCGANGeneratorImpl(int noise_size)
    : conv1_(register_module(
             "conv1",
             torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(noise_size, 256, 4)
             .bias(false)))),
      conv2_(register_module(
             "conv2",
             torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 3)
             .stride(2).padding(1).bias(false)))),
      conv3_(register_module(
             "conv3",
             torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4)
             .stride(2).padding(1).bias(false)))),
      conv4_(register_module(
             "conv4",
             torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 1, 4)
             .stride(2).padding(1).bias(false)))),
      batch_norm1_(register_module(
             "batch_norm1",
             torch::nn::BatchNorm2d(256))),
      batch_norm2_(register_module(
             "batch_norm2",
             torch::nn::BatchNorm2d(128))),
      batch_norm3_(register_module(
             "batch_norm4",
             torch::nn::BatchNorm2d(64)))
{ }

torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x)
{
    x = torch::relu(batch_norm1_->forward(conv1_->forward(x)));
    x = torch::relu(batch_norm2_->forward(conv2_->forward(x)));
    x = torch::relu(batch_norm3_->forward(conv3_->forward(x)));
    x = torch::tanh(conv4_->forward(x));
  return x;
}

