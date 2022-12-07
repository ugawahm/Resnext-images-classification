import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageTk
# Create the Tkinter window
root = tk.Tk()
root.title("Image Classifier")
root.geometry('200x200')



# Import the model class
from model import resnext50_32x4d

# Set the number of classes
num_classes = 8

# Initialize the model
model = resnext50_32x4d(num_classes=num_classes)

# Load the trained weights
model.load_state_dict(torch.load("resNext50_epoch150.pth"))

# Set the model to eval mode
model.eval()


# Create a button for selecting an image
def select_image():
  # Open a file dialog
  image_path = filedialog.askopenfilename()

  # Load the image
  image = Image.open(image_path)
  img_ = image.resize((100,100))
  img = ImageTk.PhotoImage(img_)
  lableShowImage.config(image=img)
  lableShowImage.image = img



  # Preprocess the image
  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  image = transform(image)

  # Add a batch dimension
  image = image.unsqueeze(0)

  # Move the data to the device
  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  image = image.to(device)

  # Store the image in a global variable
  select_image.image = image

# Create a button for making a prediction
def predict():
  # Check if an image has been selected
  if not hasattr(select_image, "image"):
    result_label.config(text="No image selected.")
    return

  # Make a prediction
  with torch.no_grad():
    output = model(select_image.image)
    _, predicted = torch.max(output, 1)

  # Get the class names
  class_names = ["baking_cakes", "bones", "egg_shell", "fallen_leaves", "fruits", "leftovers", "tea_leaves", "vegetable_leaves"]

  # Display the prediction results
  result_label.config(text="Prediction: " + class_names[predicted.item()])
# Create a label for the prediction results
result_label = tk.Label(root, text="")
result_label.pack()

# Create a button for selecting an image
select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack()
lableShowImage = tk.Label(root)
lableShowImage.pack()
# Create a button for making a prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

root.mainloop()

