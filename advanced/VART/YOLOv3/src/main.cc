#include <iostream>
#include "utils.hpp"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage of ADAS detection: ./yolov3 [model_file]" << std::endl;
    return -1;
  }

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
    << "yolov3 should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  auto runner = vart::Runner::create_runner(subgraph[0], "run");

  for ( int i = 0; i < runner->get_output_tensors().size(); i++) {
    printf("%s\n", runner->get_output_tensors()[i]->get_name().c_str());
  }

  runYOLO(runner.get());

  return 0;

}