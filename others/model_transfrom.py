import torch
model_weight_path = "trained_models/updata_model_110/3140.pth"  #自己的pth文件路径
out_onnx = './trained_models/onnx/SRLNet.onnx'           #保存生成的onnx文件路径
actor_critic = torch.load(model_weight_path)
# model.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu'))) #加载自己的pth文件
actor_critic.eval()

x1 = torch.randn(1, 30)
#define input and output nodes, can be customized
input_names = ["input"]
output_names = ["output"]
#convert pytorch to onnx
torch_out = torch.onnx.export(actor_critic, x1, out_onnx, export_params=True,
                              input_names=input_names, output_names=output_names)