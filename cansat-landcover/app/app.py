import spaces 
import json
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms

# --- Settings saved from the notebook (class names, tile size, grid) ---
cfg = json.load(open("config.json"))
CLASSES = cfg["classes"]
TILE = cfg["tile_px"]                                   # 100 px tiles
W, H = cfg["grid_cols"] * TILE, cfg["grid_rows"] * TILE  # photos are resized to 800x600
IMG_SIZE = cfg["input_size"]                            # 224 px model input

COLORS = {"field": (242, 193, 78), "vegetation": (46, 158, 79), "building": (214, 69, 69),
          "road": (59, 111, 212), "bare_land": (176, 137, 104), "sky": (127, 211, 232)}

# --- Load the trained model (MobileNetV3 with our 6-class head) ---
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
model.load_state_dict(torch.load("model_6class.pth", map_location="cpu"))
model.eval()

# Same preprocessing as in the notebook
prep = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
@spaces.GPU
def predict(photo, min_conf):
    """Cut the photo into tiles, classify each tile, and paint the result on the photo."""
    img = photo.convert("RGB").resize((W, H))
    boxes = [(x, y) for y in range(0, H, TILE) for x in range(0, W, TILE)]
    batch = torch.stack([prep(img.crop((x, y, x + TILE, y + TILE))) for x, y in boxes])
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
    conf, pred = probs.max(1)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    counts = {c: 0 for c in CLASSES}
    uncertain = 0
    for (x, y), p, c in zip(boxes, pred.tolist(), conf.tolist()):
        if c < min_conf:                      # not sure enough: leave the tile unpainted
            uncertain += 1
            continue
        name = CLASSES[p]
        counts[name] += 1
        draw.rectangle([x, y, x + TILE, y + TILE], fill=COLORS[name] + (120,), outline=(255, 255, 255, 200))

    result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    share = {name: n / len(boxes) for name, n in counts.items()}
    share["uncertain (hidden)"] = uncertain / len(boxes)
    return result, share

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Aerial photo"),
            gr.Slider(0.0, 0.95, value=0.0, step=0.05, label="Hide tiles when the model's confidence is below")],
    outputs=[gr.Image(label="Predicted land cover"),
             gr.Label(label="Share of tiles", num_top_classes=7)],
    title="🛰️ CanSat Land-Cover Classifier",
    description="Upload an aerial photo. It is cut into 100px tiles and each tile is coloured as field, vegetation, "
                "building, road, bare land or sky. v1: trained on EuroSAT plus 190 hand-labelled tiles from my CanSat "
                "photos, so it works best on photos like mine (low altitude, ESP32-CAM). Roads and buildings are its weak spots.",
)

demo.launch()