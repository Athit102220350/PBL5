import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch.nn.functional as F
import numpy as np

# 1️⃣ Define lại model: MobileNetV2 (không pretrained)
model = models.mobilenet_v2(pretrained=False)

# 2️⃣ Sửa lại số lớp cuối cùng: 10 classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)

# 3️⃣ Load state_dict
state_dict = torch.load("models/model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# 4️⃣ Eval mode
model.eval()

# 5️⃣ Labels
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

# Preprocess ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Nếu lúc TRAIN bạn dùng khác thì chỉnh cho đúng!
])

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check-wood', methods=['POST'])
def check_wood():
    file = request.files['image']
    img = Image.open(file).convert("RGB")
    
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    result_probs = [round(p * 100, 2) for p in sorted_probs if p >= 0.03]
    result_labels = sorted_labels[:len(result_probs)]

    response = [result_probs, result_labels]
    return jsonify(response)

@app.route('/phone')
def get_phone():
    return render_template('phone.html')

@app.route('/check-wood')
def showSignUp():
    return render_template('postImage.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
