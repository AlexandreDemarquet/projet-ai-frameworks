from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route("/")
def check_gpu():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    return jsonify({"gpu_available": gpu_available, "gpu_name": gpu_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

