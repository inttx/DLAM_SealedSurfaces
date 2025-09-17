# Sealed Surface Segmentation  

## Project Overview  
Ground sealing areas (impervious surfaces) have become a major concern in recent decades due to urbanization. Increasingly large portions of natural land are being permanently sealed by human activity, which leads to:  
- Loss of vegetation and biodiversity
- Disruption of the natural water cycle
- Urban heat islands
- Increased risks of flooding, pollution, and health issues

Quantifying and mapping sealed surfaces is therefore critical for **urban planning** and **climate adaptation strategies**.  

This project explores **deep learning methods for semantic segmentation** of sealed surfaces using remote sensing data. We compare **CNN-based** and **Transformer-based** architectures for the task, focusing on efficiency, accuracy, and generalization.  

---

## Project Goal  
Our main goal is to **train and evaluate segmentation models** that can distinguish sealed (impervious) surfaces from natural/pervious ones. To reduce compute costs, we leverage **pre-trained models** and fine-tune them for our task.  

---

## 🗂 Dataset  
- **Benchmark dataset**: [ISPRS Potsdam 2D](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
  - 37 RGB orthoimage tiles with pixel-level annotations  
  - Original 6 classes (some merged for training)  
- **Additional dataset**: Hessigheim (for testing transfer learning capabilities)  

---

## Approach  

We implement and compare two strong segmentation models:  

### DeepLabV3 (CNN-based)  
- Employs **atrous convolution** to expand receptive fields  
- Uses **Atrous Spatial Pyramid Pooling (ASPP)** to handle multi-scale segmentation  
- Provides a robust CNN baseline  

### SegFormer (Transformer-based)  
- Encoder-decoder architecture with **multi-scale Transformer backbone**  
- Efficient **overlapping patch merging** for local continuity  
- Lightweight MLP decoder for feature fusion and upsampling  
- Strong performance in both accuracy and efficiency  

---

## Training Strategy  
- Fine-tuning pre-trained models with **image augmentations** to reduce overfitting  
- **Class pooling**: sealed vs. pervious vs. rejection class (simplified setup)  
- Evaluation of training-cost vs. performance trade-offs:  
  - Initial fine-tuning of classification head/decoder  
  - Unfreezing the full model for further adaptation  
- Comparison of **full-class training** vs. **pooled-class training**  

---

## 👥 Contributors  
- Jonas Keller
- Leon Claviez
- Julian Seeger
- Janina Fritzke

---

## 📜 License  
This project is released under the **MIT License**.

