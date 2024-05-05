#ifndef ALMANAK_SIMPLEST_FC_LAYER_HPP_
#define ALMANAK_SIMPLEST_FC_LAYER_HPP_

#include <torch/torch.h>

/*
 * This network defines a layer with the specified number of input nodes ("input") and
 * a specified number of nodes in the output layer ("output").
 * Since the network consists of only one layer, the output layer is also considered the
 * first hidden layer, with the same number of nodes ("output").
 * Note : All these are fully connected nodes. (And no training portion defines here)
 */

class SimplestFCNetworkImpl : public torch::nn::Module {
public:
  /*
  * Construct the neural network architecture.
  * 
  */
  explicit SimplestFCNetworkImpl(int64_t input, int64_t output);  
  /*
  * Defines the forward computation
  * Output_matrix = (input_matrix x W(Transpose) + linear_biases) + another_bias_
  * 
  */
  torch::Tensor forward(torch::Tensor input);  
  /*
  * Returns another_bias vector, linear.weight matrix, linear.bias vector
  * 
  */
  torch::OrderedDict<std::string, torch::Tensor> show_named_parameters() const;
private:
  torch::nn::Linear first_hidden_layer_;
  torch::Tensor another_bias_;
};

#endif  // ALMANAK_SIMPLEST_FC_LAYER_HPP_

TORCH_MODULE(SimplestFCNetwork);