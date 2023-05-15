import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2
from PIL import Image
from sklearn import svm
import joblib
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Getting Cropped Lips
def getting_Lips(img, model_yolo):
  results = model_yolo(img)

  detections = results.pandas().xyxy[0]

  if detections.empty:
      return
  else:
      bboxes = detections[['xmin', 'ymin', 'xmax', 'ymax']]

      for index, row in bboxes.iterrows():
          x1, y1, x2, y2 = row
          cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

          return cropped_img
      
#Load CNN
def load_CNN():
  model = models.resnet50(pretrained=False).to(device)
  
  model.load_state_dict(torch.load('./Weights/resnet50.h5' , map_location=torch.device('cpu')) )
  model = torch.nn.Sequential(*list(model.children())[:-1])
  model.eval()

  return model


def preprocessing(img1):
   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
   data_transform = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  normalize
              ])
   
   img_tensor1 = data_transform(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))

   img1_tensor = img_tensor1.unsqueeze(0)

   model_cnn = load_CNN()

   with torch.no_grad():
      img1_features = model_cnn(img1_tensor)

   img1_vector = img1_features.squeeze().numpy()

   return img1_vector
  