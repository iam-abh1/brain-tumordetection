Brain Tumor Detection using Machine Learning
📁 Dataset
The dataset used in this project is organized into the following directory structure on Google Drive:

markdown
Copy
Edit
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
Each subfolder contains respective class images.

🚀 Features
Data loading and shuffling

Image augmentation

VGG16-based transfer learning

Training visualization

Model evaluation:

Accuracy & Loss Plot

Confusion Matrix

Classification Report

ROC Curve

Model saving and loading

New image prediction with confidence

🧱 Model Architecture
Pre-trained VGG16 (ImageNet weights, last 3 convolutional layers fine-tuned)

Flatten → Dense(128) → Dropout(0.3) → Dense(number of classes, Softmax)

🔧 Requirements
Install these in a Google Colab or local environment:

bash
Copy
Edit
pip install tensorflow pillow matplotlib scikit-learn seaborn
📌 How to Run
Mount Google Drive:

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
Set Paths:

python
Copy
Edit
train_dir = '/content/drive/MyDrive/Dataset/Training/'
test_dir = '/content/drive/MyDrive/Dataset/Testing/'
Train the Model:

The model is trained using a generator with image augmentation.

You can modify the batch size or number of epochs as needed.

Evaluate the Model:

Classification Report

Confusion Matrix

ROC Curve

Save and Load Model:

python
Copy
Edit
model.save('model.h5')
model = load_model('model.h5')
Make Predictions:

python
Copy
Edit
image_path = '/content/drive/MyDrive/Dataset/Testing/meningioma/Te-meTr_0001.jpg'
detect_and_display(image_path, model)
📊 Results
Accuracy and Loss are plotted during training.

A confusion matrix and ROC curve are generated to show model performance.

🔍 Prediction Output Example

Displays the image along with predicted tumor type and confidence score.

🧠 Class Labels
python
Copy
Edit
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
If prediction is 'notumor', it displays "No Tumor". Otherwise, it shows the tumor type and confidence.

📎 File Overview
File / Folder	Description
model.h5	Saved Keras model
detect_and_display	Predicts and displays image result
open_images	Loads and augments image datasets
datagen	Custom data generator
train_paths	List of training image paths

✨ Developed By
Kethireddy Abhilash Reddy
Kakatiya Institute Of Technology And Science
CSE(Networks)
AICTE ID:STU67fe6d529fc7f1744727378
Email ID:K.abhilashreddy0312@gmail.com
#Brain-TumorDetection

📜 License
This project is open-source and free to use for educational and research purposes.

Let me know if you'd like a requirements.txt, a Colab badge, or to auto-generate .py script files from your notebook.
