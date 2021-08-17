import onnx
# 加载模型
model = onnx.load('trained_models/onnx/SRLNet.onnx')
# 检查模型格式是否完整及正确
onnx.checker.check_model(model)
# 获取输出层，包含层名称、维度信息
output = model.graph.output
print(output)