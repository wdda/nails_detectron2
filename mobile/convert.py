import torch
import torchvision

model = torch.load('../output/model_0009999.pth')
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("app/src/main/assets/model.pt")