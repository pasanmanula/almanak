#include "simplest_fc_layer.hpp"

SimplestFCNetwork::SimplestFCNetwork(int64_t input, int64_t output)
    : first_hidden_layer_(register_module("linear", torch::nn::Linear(input, output)))
{
  another_bias_ = register_parameter("ab", torch::randn(output));
}

torch::Tensor SimplestFCNetwork::forward(torch::Tensor input)
{
  return first_hidden_layer_(input) + another_bias_;
}

torch::OrderedDict<std::string, torch::Tensor> SimplestFCNetwork::show_named_parameters() const
{
  return named_parameters();
}

