import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
from torch.nn.functional import softmax
import os


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class Xray:
    def __init__(self, models_directory="./", gpu=False):
        self.N_CLASSES = 14
        self.CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                            'Hernia']
        if gpu:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # initialize and load the model
        model = DenseNet121(self.N_CLASSES).to(device).eval()
        if gpu:
            model = torch.nn.DataParallel(model).to(device).eval()
            checkpoint = torch.load(os.path.join(models_directory, "gpu_weight.pth"))
        else:
            checkpoint = torch.load(os.path.join(models_directory, "cpu_weight.pth"), map_location=device)

        model.load_state_dict(checkpoint)

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([self.normalize(crop) for crop in crops]))
        ])

        self.model = model.to(device).eval()
        self.device = device

    def predict(self, image):
        x = self.transform(image.convert("RGB")).to(self.device)
        out = self.model(x).cpu().detach()
        probas = softmax(out).mean(0)
        ind = torch.argmax(probas).item()
        prob = probas[ind].item()
        if prob > .1:
            return {"result": 'POSITIVE', "type": self.CLASS_NAMES[ind], "probability": prob}
        else:
            return {"result": 'NEGATIVE', "type": "Healthy", "probability": 1 - probas.mean().item()}
