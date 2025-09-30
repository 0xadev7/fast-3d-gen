
module.exports = {
  apps: [
    {
      name: "generation",
      script: "app.py",
      interpreter: "python",
      args: "--host 0.0.0.0 --port 8093",
      instances: 1,
      exec_mode: "fork",
      env: {
        "HF_HOME": "/workspace/.cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
      }
    }
  ]
};
