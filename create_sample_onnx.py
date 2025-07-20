#!/usr/bin/env python3
"""
Utility script to create sample ONNX models for testing the enhanced FA-DOSA framework.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os

def create_resnet18_onnx():
    """Create a simplified ResNet-18 ONNX model."""
    # Create a simplified ResNet-18 model
    model = models.resnet18(pretrained=False)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    output_path = "onnx_models/resnet18.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Created ResNet-18 ONNX model: {output_path}")

def create_simple_conv_onnx():
    """Create a simple 2-layer CNN ONNX model."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            return x
    
    model = SimpleCNN()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 56, 56)
    
    # Export to ONNX
    output_path = "onnx_models/simple_cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Created Simple CNN ONNX model: {output_path}")

def create_bert_like_onnx():
    """Create a smaller BERT-like transformer ONNX model."""
    class SimpleBERT(nn.Module):
        def __init__(self):
            super(SimpleBERT, self).__init__()
            self.embedding = nn.Embedding(10000, 256)  # Reduced vocab size and hidden size
            self.linear1 = nn.Linear(256, 512)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(512, 256)
            self.layernorm = nn.LayerNorm(256)
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            # Flatten for linear layers
            batch_size, seq_len, hidden_size = x.shape
            x = x.view(-1, hidden_size)
            
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.layernorm(x)
            
            # Reshape back
            x = x.view(batch_size, seq_len, hidden_size)
            return x
    
    model = SimpleBERT()
    model.eval()
    
    # Create dummy input with smaller sequence length
    dummy_input = torch.randint(0, 10000, (1, 64))
    
    # Export to ONNX
    output_path = "onnx_models/bert_small.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['output']
    )
    print(f"Created smaller BERT-like ONNX model: {output_path}")

if __name__ == "__main__":
    # Ensure onnx_models directory exists
    os.makedirs("onnx_models", exist_ok=True)
    
    print("Creating sample ONNX models...")
    
    try:
        create_simple_conv_onnx()
        create_resnet18_onnx()
        create_bert_like_onnx()
        print("\nAll sample ONNX models created successfully!")
        print("Available models:")
        print("- simple_cnn: 2-layer CNN with Conv->ReLU patterns")
        print("- resnet18: Standard ResNet-18 architecture")
        print("- bert_small: Smaller BERT-like transformer with linear layers (under 100MB)")
    except Exception as e:
        print(f"Error creating ONNX models: {e}")
        print("Make sure PyTorch and torchvision are installed.")