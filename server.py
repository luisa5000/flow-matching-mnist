"""
HTTP server that generates a digit image on each request.

Usage:
    python server.py --checkpoint path/to/model.ckpt [--port 5000] [--num_steps 10]

Request:
    GET /generate?digit=3

Response:
    PNG image of the requested digit.
    400 if digit is missing or out of range [0-9].
    400 if the model is unconditional and a digit is requested.
"""

import argparse
import io
import torch
from flask import Flask, request, send_file, jsonify
from torchvision.utils import save_image

from models.fm import ImageFlowMatcher
from models.class_cond_fm import ImageFlowMatcherClassCond


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "num_classes" in ckpt.get("hyper_parameters", {}):
        return ImageFlowMatcherClassCond.load_from_checkpoint(checkpoint_path, map_location=device)
    return ImageFlowMatcher.load_from_checkpoint(checkpoint_path, map_location=device)


def create_app(checkpoint: str, num_steps: int) -> Flask:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint}")
    model = load_model(checkpoint, device)
    model.to(device)
    model.eval()
    is_conditional = isinstance(model, ImageFlowMatcherClassCond)
    print(f"Model type: {'class-conditional' if is_conditional else 'unconditional'}")

    app = Flask(__name__)

    @app.get("/generate")
    def generate():
        digit_str = request.args.get("digit")
        if digit_str is None:
            return jsonify(error="'digit' query parameter is required"), 400
        try:
            digit = int(digit_str)
        except ValueError:
            return jsonify(error="'digit' must be an integer"), 400
        if digit < 0 or digit > 9:
            return jsonify(error="'digit' must be between 0 and 9"), 400
        if not is_conditional:
            return jsonify(error="loaded checkpoint is unconditional; digit selection is not supported"), 400

        label = torch.tensor([digit], device=device)
        with torch.no_grad():
            image = model.generate(batch_size=1, num_steps=num_steps, y=label)

        buf = io.BytesIO()
        save_image(image[0], buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    return app


def main():
    parser = argparse.ArgumentParser(description="Digit generation server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--num_steps", type=int, default=10, help="ODE solver steps")
    args = parser.parse_args()

    app = create_app(args.checkpoint, args.num_steps)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
