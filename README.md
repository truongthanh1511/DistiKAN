# DistiKAN

Submitting to Vietnam Journal of Computer Science. 
Framework Name: DistiKAN
Title: Distillation-Enhanced KAN for Efficient Fine-Grained Intangible Cultural Heritage Recognition
Abstract:
Fine-grained intangible cultural heritage recognition remains challenging because many classes are distinguished only by subtle visual cues. Practical applications also require low-latency inference in resource-limited settings. These constraints create a difficult trade-off between recognition accuracy and inference efficiency. We propose DistiKAN, a teacher-student framework that combines a lightweight backbone, a spline-based Kolmogorov-Arnold Network classifier, and response-based knowledge distillation. The key idea is to add flexibility only at the final decision stage, while teacher guidance helps stabilize training of the adaptive student head. On VN-ICH, a Vietnamese intangible cultural heritage benchmark, the best compact student configuration achieves 94.25\% accuracy and 94.03\% Macro-F1 with 6.5 ms latency per image. Ablation results show improvements over matched baselines and indicate that distillation generally makes these gains more reliable in compact students. On two external benchmarks, the method improves Macro-F1 by up to 3.73 points, supporting its effectiveness beyond the target cultural dataset.
