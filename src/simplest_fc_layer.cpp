#include "simplest_fc_layer.hpp"

SimplestFCNetworkImpl::SimplestFCNetworkImpl(int64_t input, int64_t output)
    : first_hidden_layer_(register_module("first_hidden_linear", torch::nn::Linear(input, output)))
{
  another_bias_ = register_parameter("ab", torch::randn(output));
}

torch::Tensor SimplestFCNetworkImpl::forward(torch::Tensor input)
{
  return first_hidden_layer_(input) + another_bias_;
}

torch::OrderedDict<std::string, torch::Tensor> SimplestFCNetworkImpl::show_named_parameters() const
{
  return named_parameters();
}

