import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import model_utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DISTILL_FOLDER'] = 'distillation_runs'

# Tạo thư mục upload nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Biến toàn cục để cache model (tránh load lại mỗi lần request)
CURRENT_MODELS = {}
CURRENT_CLASS_NAMES = []
CURRENT_EXP_ID = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/')
def index():
    experiments = model_utils.scan_experiments(app.config['DISTILL_FOLDER'])
    return render_template('index.html', experiments=experiments)

@app.route('/load_experiment', methods=['POST'])
def load_experiment():
    global CURRENT_MODELS, CURRENT_CLASS_NAMES, CURRENT_EXP_ID
    
    exp_id = request.form.get('exp_id')
    exp_path = os.path.join(app.config['DISTILL_FOLDER'], exp_id)
    
    if not os.path.exists(exp_path):
        return jsonify({"status": "error", "message": "Experiment not found"}), 404
        
    try:
        # Load models into memory
        CURRENT_MODELS, CURRENT_CLASS_NAMES = model_utils.load_models(exp_path, DEVICE)
        CURRENT_EXP_ID = exp_id
        
        model_list = list(CURRENT_MODELS.keys())
        return jsonify({
            "status": "success", 
            "message": f"Loaded models: {', '.join(model_list)}",
            "models": model_list,
            "classes": CURRENT_CLASS_NAMES
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not CURRENT_MODELS:
        return jsonify({"status": "error", "message": "Please select an experiment first"}), 400
        
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    # Lưu file tạm
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Preprocess ảnh
        img = Image.open(filepath).convert('RGB')
        transform = model_utils.get_transform()
        img_tensor = transform(img).unsqueeze(0) # Add batch dimension
        
        results = {}
        
        # Chạy dự đoán cho tất cả model đã load
        for model_type, model in CURRENT_MODELS.items():
            res = model_utils.predict_image(model, img_tensor, DEVICE, CURRENT_CLASS_NAMES)
            results[model_type] = res
            
        return jsonify({
            "status": "success",
            "image_url": filepath,
            "results": results
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/evaluate_folder', methods=['POST'])
def evaluate_folder():
    # Chức năng nâng cao: Đánh giá cả thư mục
    # Để đơn giản hóa, API này nhận đường dẫn thư mục ảnh từ client
    # (Lưu ý: Trong môi trường web thực tế, browser không gửi full path vì bảo mật, 
    # nên thường ta sẽ unzip file upload hoặc nhập đường dẫn server)
    
    folder_path = request.form.get('folder_path')
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({"status": "error", "message": "Invalid folder path"}), 400
    
    if not CURRENT_MODELS:
        return jsonify({"status": "error", "message": "Models not loaded"}), 400

    summary = {m_type: {"total_time": 0, "count": 0} for m_type in CURRENT_MODELS}
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        return jsonify({"status": "error", "message": "No images found"}), 400
        
    # Limit số lượng ảnh để test nhanh (ví dụ 50 ảnh)
    image_files = image_files[:50] 
    
    transform = model_utils.get_transform()
    
    for fname in image_files:
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            for m_type, model in CURRENT_MODELS.items():
                res = model_utils.predict_image(model, img_tensor, DEVICE, CURRENT_CLASS_NAMES)
                summary[m_type]["total_time"] += res["time_ms"]
                summary[m_type]["count"] += 1
        except:
            continue
            
    # Tính trung bình
    final_stats = {}
    for m_type, stats in summary.items():
        if stats["count"] > 0:
            final_stats[m_type] = {
                "avg_time_ms": round(stats["total_time"] / stats["count"], 2),
                "samples": stats["count"]
            }
            
    return jsonify({
        "status": "success",
        "stats": final_stats
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)