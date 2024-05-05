/*
 * Copyright (c) no body lol
 *
 */
#include <iostream>
#include "simplest_fc_layer.hpp"

void execute_simple_fc_example()
{
  // Defined the neural network object in the stack
  SimplestFCNetwork NN(4, 5);
  // Forward pass on randomly generated weights and biases.
  torch::Tensor results = NN->forward(torch::ones({2,4}));
  // Forward pass result
  std::cout << results <<std::endl;
  // Show randomly generated weights and biases.
  for (const auto& pair : NN->show_named_parameters()) {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
}

int main()
{
  std::cout << "Hello I'm Almanak" << std::endl;
  execute_simple_fc_example();
  return 0;
}