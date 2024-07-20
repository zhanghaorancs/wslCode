import torch.distributed.rpc as rpc
import os

os.environ["MASTER_ADDR"] = "172.27.5.34"
os.environ["MASTER_PORT"] = "29500"

def hello():
    return "hello world!"

if __name__ == "__main__":
    print("11111111")
    rpc.init_rpc(
        name="edge17",
        rank=0,
        world_size=2,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            # rpc_timeout=1000,
            init_method="tcp://172.27.5.34:29500",
        ),
    )

    print("server init complete")   
    res = rpc.rpc_sync("edge17",hello)
    print(res)
    print("rpc_sync complete") 

    rpc.shutdown()