import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

FILE = 'model_cpu.pth'
#Get device cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.c1 = nn.Conv2d(1,12,(5,5),stride=1,padding=0)
        self.c2 = nn.Conv2d(12,24,(3,3),stride=1,padding=0)
        self.c3 = nn.Conv2d(24,48,(3,3),stride=1,padding=0)
        self.l1 = nn.Linear(48*5*5,400)
        self.l2 = nn.Linear(400,100)
        self.l3 = nn.Linear(100,43)

    def forward(self,x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.c3(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,48*5*5)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return F.log_softmax(x,dim=1)


#Load saved model state
loaded_model = Net()
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.to(device)
loaded_model.eval()

#Get input image
img=cv2.imread('image.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (56, 56)) 

#Show the image
cv2.imshow('Input 56*56',img)
cv2.waitKey(0)

#Convert np.array to tensor
in_x = torch.from_numpy(img)
if len(in_x.shape)==2:
    in_x=in_x.unsqueeze(0).unsqueeze(0) #Get shape of [1,1,56,56]

in_x = in_x.type(torch.float)

#Print input shape
#print(in_x.shape)

#Get the output tensor
out_y = loaded_model(in_x)
#print('OUTPUTS: ',out_y)

#Get prediction
Y_max=out_y.argmax(dim=1, keepdim=True)[0][0]
#print('Prediction: ',Y_max)

class_dict={ 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 
            10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 
            14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 
            18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 
            21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 
            39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'}

print(class_dict[int(Y_max.numpy())])