import torch
import time

encoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)


#batch_null = torch.zeros((64, 3, 256, 256))
batch_one = torch.rand((64, 3, 256, 256))

tic = time.perf_counter()
encoder(batch_one)
toc = time.perf_counter()
print(f"Infered in {toc - tic:0.4f} seconds")



