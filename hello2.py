from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from keras.models import load_model
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)

# 🔹 Model 1: Kiểm tra có phải là gỗ không (Keras)
model_check = load_model('models/model_prev.keras')

# 🔹 Model 2: Dự đoán loại gỗ (PyTorch - MobileNetV2)
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
model.load_state_dict(torch.load('models/model.pth', map_location='cpu'))
model.eval()

# 🔹 Transform cho model 2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 🔹 Danh sách loại gỗ
labels = [
    "Ailanthus altissima - Gỗ tiêu huyền",
    "Alnus glutinosa - Gỗ tống quán sủi",
    "Castanea sativa - Gỗ dẻ ngựa",
    "Fagus sylvatica - Gỗ sồi dẻ gai châu Âu",
    "Fraxinus ornus - Gỗ tần bì hoa",
    "Juglans regia - Gỗ óc chó",
    "Picea abies - Gỗ vân sam Na Uy",
    "Pinus sylvestris - Gỗ thông rừng",
    "Quercus cerris - Gỗ sồi Thổ Nhĩ Kỳ",
    "Robinia pseudoacacia - Gỗ keo đen"
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/phone')
def get_phone():
    return render_template('phone.html')

@app.route('/check-wood')
def showSignUp():
    return render_template('postImage.html')

@app.route('/check-wood', methods=['POST'])
def check_wood():
    file = request.files['image']
    img = Image.open(file).convert("RGB")

    # ⭐ Bước 1: Model 1 - Kiểm tra có phải là gỗ không
    img_prev = img.resize((224, 224))
    img_prev_array = np.array(img_prev) / 255.0
    img_prev_array = np.expand_dims(img_prev_array, axis=0)

    prediction_prev = model_check.predict(img_prev_array)[0]
    predicted_class = np.argmax(prediction_prev)

    # Nếu không phải gỗ
    if predicted_class != 2:
        return jsonify("Not wood")

    # ⭐ Bước 2: Model 2 - Dự đoán loại gỗ
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = [round(probs[i] * 100, 2) for i in sorted_indices if probs[i] >= 0.03]
    sorted_labels = [labels[i] for i in sorted_indices[:len(sorted_probs)]]

    return jsonify([sorted_probs, sorted_labels])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
