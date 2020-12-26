#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <iostream>

using namespace std;
using namespace tensorflow;


int main()
{
    // set up input paths
    const string graph_file = "/home/chenwei/HDD/Project/Complex-YOLOv2/checkpoints/model.ckpt-18.meta";
    const string checkpoints_file = "/home/chenwei/HDD/Project/Complex-YOLOv2/checkpoints/model.ckpt-18";

    // create session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        cout << status.ToString() << "\n";
        throw runtime_error("Could not create Tensorflow session.");
    }

    // read graph in the protobuf
    MetaGraphDef graph_def;
    Status status_od_net = ReadBinaryProto(Env::Default(), graph_file, &graph_def);
    if (!status_od_net.ok())
    {
        throw std::runtime_error("Error reading graph definition from " + graph_file + ": " + status_od_net.ToString());
    }

    // add the graph to the session
    status_od_net = session->Create(graph_def.graph_def());

    // read weights from the saved checkpoints
    Tensor checkpoint_path_tensor(DT_STRING, TensorShape());
    checkpoint_path_tensor.scalar<std::string>()() = checkpoints_file;
    status_od_net = session->Run(
            {{graph_def.saver_def().filename_tensor_name(), checkpoint_path_tensor}, },
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);




    return 0;
}