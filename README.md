# Deep Neural Network for classification of H&E-stained Colorectal Polyps - Exploring the Pipeline of Computer-Assisted Histopathology

Fine tuned on top of the pretreined model by [Ciga et al. (2022)](https://github.com/ozanciga/self-supervised-histopathology).
Read the paper here:

Starter code:

```
import torch
import torchvision

MODEL_PATH = 'final_model1.pt'
NUM_CLASSES = 1

# Get gpu, mps or cpu device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load pretrained model
model = torchvision.models.__dict__['resnet18'](weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
model = model.to(device)

# Example inference
imgs = torch.rand((10, 3, 224, 224), device=device)

model.eval()
logits = model(imgs)
```

### Citation
If you find this work useful or

```
@mastersthesis{Brunzell2024,
  author = {Stina Brunzell},
  title = {Deep Neural Network for classification of H&E-stained Colorectal Polyps 
            - Exploring the Pipeline of Computer-Assisted Histopathology},
  school = {Uppsala University},
  year = {2024},
  url = {}
}
```
