import torch
from torchvision.transforms import transforms
from torchvision.models import resnet18
from torch.nn.functional import softmax
import os

from .DenseNet import DenseNet121
from .EffNet import EfficientNet

class Xray:
    def __init__(self, gpu=False):
        models_directory = os.path.dirname(os.path.abspath(__file__))
        # DENSENET
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
        model_dense = DenseNet121(self.N_CLASSES).to(device).eval()
        if gpu:
            model_dense = torch.nn.DataParallel(model_dense).to(device).eval()
            checkpoint = torch.load(os.path.join(models_directory, "gpu_weight.pth"))
        else:
            checkpoint = torch.load(os.path.join(models_directory, "cpu_weight.pth"), map_location=device)

        model_dense.load_state_dict(checkpoint)

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        self.transform_dense = transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([self.normalize(crop) for crop in crops]))
        ])

        self.model_dense = model_dense.to(device).eval()
        self.device = device

        # EFFNET
        model_eff = EfficientNet.from_name(model_name="efficientnet-b0",
                                           params=[1.0, 1.0, 224, 0.2],
                                           override_params={'num_classes': 2})
        state_dict = torch.load(os.path.join(models_directory, "effnet_weight.pth"), map_location=device)
        model_eff.load_state_dict(state_dict)

        self.model_eff = model_eff.to(device).eval()

        self.transform_eff = transforms.Compose([ transforms.Resize(224),
                                                  transforms.Grayscale(3),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
        #Resnet for detecting xray out of random images
        resnet_state_dict = torch.load(os.path.join(models_directory, "nonxray.pth"), map_location=device)
        model_resnet = resnet18()
        model_resnet.fc = torch.nn.Linear(model_resnet.fc.in_features, 2)
        model_resnet.load_state_dict(resnet_state_dict)

        self.model_resnet = model_resnet.to(device).eval()

        self.transform_resnet = transforms.Compose([transforms.Resize(100),
                                                    transforms.CenterCrop(64),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                    ])

    def predict_dense(self, image):
        x = self.transform_dense(image).to(self.device)
        out = self.model_dense(x).cpu().detach()
        probas = softmax(out*5).mean(0)
        return list(probas.numpy()*100)

    def predcit_eff(self, image):
        x = self.transform_eff(image).unsqueeze(0).to(self.device)
        out = self.model_eff(x).cpu().detach()
        proba = softmax(out)[0, 0].item()
        return proba*100

    def predict_resnet(self, image):
        x = self.transform_resnet(image).unsqueeze(0).to(self.device)
        out = self.model_resnet(x).cpu().detach()
        return torch.argmax(out, dim=1).item()

    def predict(self, image):
        image = image.convert("RGB")
        is_xray = self.predict_resnet(image)
        if not is_xray:
            return {"result": "RANDOM"}
        healthy_proba = self.predcit_eff(image)
        disease_proba = self.predict_dense(image)
        disease_proba = list(zip(self.CLASS_NAMES, disease_proba))
        disease_proba.sort(key=lambda x: x[1], reverse=True)

        if healthy_proba > 50:
            result = "NEGATIVE"
            _type = "Healthy"
        else:
            result = "POSITIVE"
            _type = "Not Healthy"
            healthy_proba = 100 - healthy_proba

        r = {"result":result,
             "type":_type,
             "probability": healthy_proba,
             "condition similarity rate": disease_proba}
        return r

