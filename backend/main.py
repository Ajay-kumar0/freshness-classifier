from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import io

app = FastAPI(title="Freshness Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("freshness_model.pt", map_location=device)
class_names = checkpoint["class_names"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return {
        "label": class_names[idx.item()],
        "confidence": round(conf.item(), 4),
        "all_probs": {class_names[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }

@app.get("/health")
def health():
    return {"status": "ok"}