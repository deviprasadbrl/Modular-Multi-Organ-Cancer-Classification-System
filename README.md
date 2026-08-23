# Modular-Multi-Organ-Cancer-Classification-System

A modular deep-learning system that uses a router model to identify the medical imaging domain and then sends the image to a specialized expert model for organ-specific classification trained on that specific organ.

## Project Overview

This project implements a modular deep-learning system for medical image classification. Instead of using a single model to classify images from different medical domains, the system uses a two-stage architecture:

1. A **Router Model** first identifies the domain of the input image.
2. The image is then passed to a specialized **Expert Model** trained specifically for that domain.

The current system contains three expert models:

- 🧠 **Brain Expert** — classifies brain MRI images into four categories.
- 🩻 **Kidney Expert** — classifies kidney CT images into four categories.
- 🔬 **Oral Expert** — classifies oral histopathology images into two categories.

The router also contains a fourth **Random** category to identify images outside the supported domains.

### High-Level Flow

Input Image  
↓  
Router Model  
↓  
Domain Selection  
↓  
Specialized Expert Model  
↓  
Final Classification  
↓  
Grad-CAM Visualization
