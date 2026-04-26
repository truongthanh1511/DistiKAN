import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
# Giả sử người dùng đã cài fast-kan như trong notebook
# Nếu không, cần copy source code FastKAN vào đây
try:
    from fastkan import FastKAN
except ImportError:
    print("Warning: 'fastkan' library not found. Distillation model might fail to load if it uses KAN.")
    # Dummy placeholder nếu không có thư viện (để code không crash ngay lập tức)
    class FastKAN(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()

# --- ĐỊNH NGHĨA LỚP MÔ HÌNH (Lấy từ notebook) ---
class FastKANClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, 
                 kan_hidden: int, kan_grids: int = 8, dropout: float = 0.1):
        super().__init__()
        # Backbone pretrained = False vì ta sẽ load weights
        self.backbone = timm.create_model(
            model_name,
            pretrained=False, 
            num_classes=0,
            global_pool=""
        )
        self.feature_dim = self.backbone.num_features
        
        self.flatten = nn.Flatten()
        self.bn_adapter = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FastKAN classifier
        self.classifier = FastKAN(
            layers_hidden=[self.feature_dim, kan_hidden, num_classes],
            num_grids=kan_grids,
            grid_min=-3.0,
            grid_max=3.0,
            use_base_update=True,
            spline_weight_init_scale=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        if x.dim() == 4: 
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = self.flatten(x)
        elif x.dim() == 3:
            x = x.mean(dim=1)
        x = self.bn_adapter(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# --- HELPER FUNCTIONS ---

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def scan_experiments(root_dir="distillation_runs"):
    """Quét thư mục để tìm các thí nghiệm hợp lệ"""
    experiments = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        return experiments

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        report_path = os.path.join(folder_path, "experiment_report.json")
        
        if os.path.isdir(folder_path) and os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Kiểm tra file weights
                baseline_path = os.path.join(folder_path, "baseline", "baseline_weights.pth")
                distill_path = os.path.join(folder_path, "distillation", "distillation_weights.pth")
                
                experiments.append({
                    "id": folder_name,
                    "name": f"{config.get('teacher_name')} -> {config.get('student_name')}",
                    "path": folder_path,
                    "config": config,
                    "has_baseline": os.path.exists(baseline_path),
                    "has_distill": os.path.exists(distill_path)
                })
            except Exception as e:
                print(f"Error reading {folder_name}: {e}")
    return experiments

def load_models(exp_path, device='cpu'):
    """Load model baseline và distillation dựa trên config"""
    report_path = os.path.join(exp_path, "experiment_report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    student_name = report['student_name']
    num_classes = report['num_classes']
    class_names = report['class_names']
    
    # Lấy config distillation (nếu có) để biết tham số KAN
    kan_hidden = 128 # Default fallback
    kan_grids = 8    # Default fallback
    
    if 'distillation_results' in report and 'config' in report['distillation_results']:
        d_conf = report['distillation_results']['config']
        kan_hidden = d_conf.get('kan_hidden', 128)
        kan_grids = d_conf.get('kan_grids', 8)

    models = {}

    # 1. Load Baseline (Thường là timm thuần)
    baseline_weight_path = os.path.join(exp_path, "baseline", "baseline_weights.pth")
    if os.path.exists(baseline_weight_path):
        try:
            model = timm.create_model(student_name, pretrained=False, num_classes=num_classes)
            state_dict = torch.load(baseline_weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False) # strict=False để tránh lỗi nhỏ nếu header khác biệt
            model.to(device)
            model.eval()
            models['baseline'] = model
        except Exception as e:
            print(f"Failed to load baseline: {e}")

    # 2. Load Distillation (Student với KAN)
    distill_weight_path = os.path.join(exp_path, "distillation", "distillation_weights.pth")
    if os.path.exists(distill_weight_path):
        try:
            # Sử dụng class FastKANClassifier đã định nghĩa ở trên
            model = FastKANClassifier(student_name, num_classes, kan_hidden, kan_grids)
            state_dict = torch.load(distill_weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            models['distillation'] = model
        except Exception as e:
            print(f"Failed to load distillation: {e}")

    return models, class_names

def predict_image(model, image_tensor, device, class_names):
    """Dự đoán một ảnh và đo thời gian"""
    image_tensor = image_tensor.to(device)
    
    # Warmup (optional)
    # _ = model(image_tensor)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000 # ms
    
    top_prob, top_idx = torch.max(probabilities, 1)
    top_prob = top_prob.item()
    top_class = class_names[top_idx.item()]
    
    return {
        "class": top_class,
        "confidence": round(top_prob * 100, 2),
        "time_ms": round(inference_time, 2),
        "all_probs": probabilities.cpu().numpy()[0].tolist()
    }