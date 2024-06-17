# Deep Neural Network for Classification of H&E-stained Colorectal Polyps - Exploring the Pipeline of Computer-Assisted Histopathology

Abstract:
Colorectal cancer is one of the most prevalent malignancies globally and recently introduced digital pathology enables the use of machine learning as an aid for fast diagnostics. This project aimed to develop a deep neural network model to specifically identify and differentiate dysplasia in the epithelium of colorectal polyps and was posed as a binary classification problem. The available dataset consisted of 80 whole slide images of different H\&E-stained polyp sections, which were parted info smaller patches, annotated by a pathologist. The best performing model was a pretrained ResNet-18 by [Ciga et al. (2022)](https://github.com/ozanciga/self-supervised-histopathology), utilising a weighted sampler, weight decay and augmentation during fine tuning. Reaching an area under precision-recall curve of 0.9989 and 97.41\% accuracy on previously unseen data, the model's performance was determined to slightly underperform compared to the task's intra-observer variability and be in alignment with the inter-observer variability. Read the paper here:

You can download the final model's state dict with the **.pt** extension here. It was trained using the labels 0 - all tissue in the patch is normal, and 1 - some abnormal tissue is present in the patch.

Starter code:

```
import torch
import torchvision

MODEL_PATH = 'final_model1.pt'
NUM_CLASSES = 1

# Get gpu, mps or cpu device
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

### Reference
If you find this work useful, please use below citation.

```
@mastersthesis{stinabr2024,
  author = {Stina Brunzell},
  title = {Deep Neural Network for Classification of H\&E-stained Colorectal Polyps - Exploring the Pipeline of Computer-Assisted Histopathology},
  school = {Uppsala University},
  year = {2024},
  url = {}
}
```
