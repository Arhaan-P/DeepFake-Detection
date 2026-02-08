# II. LITERATURE REVIEW

This section reviews prior work across three related domains: skeleton-based gait recognition, facial deepfake detection, and motion-based forgery detection. We identify a critical research gap at the intersection of gait analysis and deepfake detection, which our proposed system addresses.

---

## A. Skeleton-Based Gait Recognition

Gait recognition aims to identify individuals by their walking patterns and has seen significant advances through deep learning. While our work targets deepfake detection rather than person identification, gait recognition establishes that walking patterns are highly discriminative biometric features — a property we exploit for forgery detection.

**Graph Convolutional Approaches.** Teepe et al. [1] introduced GaitGraph, the first framework to apply Graph Convolutional Networks (GCNs) directly to skeleton pose sequences for gait recognition. By modeling the topological relationships between body joints explicitly, GaitGraph avoids reliance on appearance-based silhouette features that are sensitive to clothing and carrying conditions. The method achieved state-of-the-art model-based performance on the CASIA-B dataset (124 subjects). Fu et al. [2] extended this paradigm with GPGait, a Generalized Pose-based Gait recognition framework incorporating Human-Oriented Transformation (HOT) and Part-Aware Graph Convolutional Networks (PAGCN). GPGait demonstrated superior cross-dataset generalization across CASIA-B, OUMVLP-Pose, Gait3D, and GREW benchmarks, achieving results comparable to silhouette-based methods while maintaining robustness to viewpoint and clothing changes.

**Transformer-Based Architectures.** The application of transformer architectures has yielded significant improvements, particularly on unconstrained datasets. Catruna et al. [3] proposed GaitPT (Gait Pyramid Transformer), which processes skeleton sequences as flattened token inputs through a hierarchical transformer, eliminating the need for hand-crafted graph topologies. GaitPT achieved 82.6% accuracy on CASIA-B and 52.16% Rank-1 on the challenging GREW dataset (4,000 subjects, unconstrained walking), outperforming both skeleton-based and appearance-based methods in the wild. Cosma and Radoi [4] developed GaitFormer, achieving 92.5% identification accuracy on CASIA-B through zero-shot transfer from DenseGait pretraining — a 14.2% improvement over comparable skeleton methods. More recently, Sheng et al. [5] proposed GaitTriViT and GaitVViT, incorporating parallel spatial-temporal branches using Vision Transformer blocks alongside temporal global attention mechanisms.

**Hybrid and Specialized Approaches.** Zhang et al. [6] combined Spatial Transformer Networks with Temporal Convolutional Networks in Gait-TR, maintaining ~90% accuracy even in challenging walking-with-coats scenarios. PoseMapGait [7] introduced a dual-stream architecture fusing CNN-based heatmap evolution features with GCN skeleton features, achieving state-of-the-art model-based performance on CASIA-B. SPOSGait [8] applied neural architecture search to gait recognition for the first time, discovering architectures that achieved state-of-the-art results across CASIA-B, OUMVLP, Gait3D, and GREW simultaneously.

A key observation from gait recognition literature is the dramatic performance gap between controlled and unconstrained settings. Methods achieving 80–96% accuracy on CASIA-B drop to 40–52% on GREW [9], indicating that gait analysis in real-world conditions remains an open challenge.

**TABLE I: SKELETON-BASED GAIT RECOGNITION METHODS**

| Method          | Year | Architecture              | Dataset                       | Subjects | Accuracy / Rank-1        | Key Contribution                   |
| --------------- | ---- | ------------------------- | ----------------------------- | -------- | ------------------------ | ---------------------------------- |
| GaitGraph [1]   | 2021 | GCN                       | CASIA-B                       | 124      | SOTA (model-based)       | First skeleton-only GCN for gait   |
| GPGait [2]      | 2023 | PAGCN                     | CASIA-B, OUMVLP, Gait3D, GREW | 124–10K  | Comparable to silhouette | Cross-dataset generalization       |
| GaitPT [3]      | 2023 | Pyramid Transformer       | CASIA-B / GREW                | 124 / 4K | 82.6% / 52.16%           | Best skeleton-based on GREW        |
| GaitFormer [4]  | 2023 | Vision Transformer        | CASIA-B / FVG                 | 124      | 92.5% / 85.33%           | Zero-shot DenseGait transfer       |
| GaitTriViT [5]  | 2025 | ViT + Temporal Attn       | CASIA-B, OUMVLP               | 124–10K  | SOTA                     | Parallel spatial-temporal branches |
| Gait-TR [6]     | 2023 | Spatial Transformer + TCN | CASIA-B                       | 124      | ~90% (with coats)        | Robust to clothing variation       |
| PoseMapGait [7] | 2023 | Dual-stream CNN+GCN       | CASIA-B, CMU-MoB              | 124      | SOTA (model-based)       | Heatmap + skeleton fusion          |
| SPOSGait [8]    | 2022 | NAS-discovered            | CASIA-B, OUMVLP, Gait3D, GREW | 124–10K  | SOTA (all benchmarks)    | First NAS for gait recognition     |

---

## B. Facial Deepfake Detection

Facial deepfake detection identifies manipulated face content in images and videos. While our approach differs fundamentally by analyzing gait rather than facial artifacts, these methods represent the dominant paradigm against which novel approaches are contextualized.

**Spatial Artifact Detection.** Zhao et al. [10] reformulated deepfake detection as a fine-grained classification problem, introducing Multi-Attentional Deepfake Detection with multiple spatial attention heads to localize subtle facial artifacts. The architecture incorporates textural feature enhancement and attention-guided data augmentation, achieving near-perfect AUC on FaceForensics++ (FF++). Cao et al. [11] proposed RECCE (Reconstruction-Classification Learning), which models only genuine face distributions rather than characterizing specific forgery methods. By learning to reconstruct real faces while failing on forgeries, RECCE achieves superior cross-dataset generalization: 64.31% AUC on WildDeepfake when trained on Celeb-DF, exceeding prior best by 4.57%.

**Temporal Coherence Approaches.** Zheng et al. [12] developed FTCN (Fully Temporal Convolution Network), which focuses exclusively on temporal modeling by reducing spatial convolution kernels to size 1. Combined with a Temporal Transformer for long-range dependencies, FTCN achieves 99.7% generalization accuracy to novel forgery types in leave-one-out evaluation on FF++. This result highlights that temporal incoherence — the inability of synthesis models to maintain frame-to-frame consistency — may be more generalizable than spatial artifacts.

**Spatio-Temporal Hybrid Methods.** Wang et al. [13] introduced AltFreezing, a training strategy that alternately freezes spatial and temporal weights of a 3D ConvNet, forcing the network to learn both artifact types independently. AltFreezing achieves 96.7% average AUC across multiple datasets and 89.5% on the challenging Celeb-DF benchmark, with exceptional robustness to compression artifacts. Recent work on Frequency-Enhanced Self-Blended Images (FSBI) [14] combines self-blended training data with Discrete Wavelet Transform features through an EfficientNet-B5 backbone, achieving state-of-the-art cross-dataset performance.

**CNN-LSTM-Transformer Hybrids for Face Detection.** Notably, hybrid architectures combining CNNs, LSTMs, and Transformers have been applied to facial deepfake detection [15], achieving 98% AUC and 90.6% F1-score on benchmark datasets while demonstrating robustness to video quality degradation. This architectural paradigm parallels our approach, though we apply it to gait features rather than facial frames.

A persistent challenge in facial deepfake detection is cross-dataset generalization. Models trained on FaceForensics++ often achieve near-perfect in-domain performance but degrade substantially on independent datasets [11][13]. The Deepfake-Eval-2024 benchmark [16] reveals that peak detector accuracy remains below 90%, falling short of human analyst performance, indicating that purely face-based approaches have fundamental limitations.

**TABLE II: FACIAL DEEPFAKE DETECTION METHODS**

| Method                     | Year | Architecture                  | Dataset                      | Key Metric           | Key Contribution                                      |
| -------------------------- | ---- | ----------------------------- | ---------------------------- | -------------------- | ----------------------------------------------------- |
| Multi-Attentional [10]     | 2021 | VGG + Multi-Attention         | FF++                         | AUC ~99%             | Fine-grained classification with textural enhancement |
| RECCE [11]                 | 2022 | Reconstruction-Classification | Celeb-DF, WildDeepfake, DFDC | AUC 64.31% (cross)   | Models real faces only; best generalization           |
| FTCN [12]                  | 2021 | Temporal Conv + Transformer   | FF++ (cross-forgery)         | 99.7% generalization | Temporal-only; best cross-forgery transfer            |
| AltFreezing [13]           | 2023 | 3D ConvNet (alt-frozen)       | Celeb-DF, FF++, DFDC         | AUC 96.7% (avg)      | Alternating spatial/temporal weight freezing          |
| FSBI [14]                  | 2024 | EfficientNet-B5 + DWT         | FF++, Celeb-DF               | SOTA (cross-dataset) | Frequency domain + self-blended images                |
| Hybrid CNN-LSTM-Trans [15] | 2024 | CNN+BiLSTM+Transformer        | FF++, Celeb-DF               | AUC 98%, F1 90.6%    | Robust to compression; parallel temporal paths        |

---

## C. Motion-Based and Body-Level Deepfake Detection

While facial deepfake detection has received extensive attention, approaches leveraging body movement, pose estimation, or gait analysis for forgery detection remain substantially underexplored. This category represents the most directly relevant prior work to our research.

**Pose-Based Motion Analysis.** Recent work [17] proposes analyzing human movement patterns through pose estimation and LSTM networks to detect unnatural transitions that AI synthesis models struggle to replicate. The methodology extracts skeletal motion data, normalizes frames to uniform dimensions, and feeds pose sequences into LSTMs that learn features such as joint displacement, velocity, and acceleration. Preliminary results report approximately 93% accuracy, outperforming conventional CNN-based image detection and demonstrating that motion inconsistencies are effective forgery indicators. However, this work employs generic pose features without specialized gait analysis — it does not extract biomechanically meaningful features such as joint angles, stride symmetry, or normalized gait landmarks.

**Forensic Gait Biometrics.** The reliability of gait as a biometric identifier has been demonstrated in forensic applications. In a landmark 2025 case, gait and body structure biometrics were accepted as court evidence in a European Union murder case [18]. When facial recognition proved impractical due to poor resolution (face-to-5-pixel at 30m distance), gait biometrics using 31 parameters successfully identified the perpetrator with an Equal Error Rate below 0.1% [18]. This legal precedent validates that gait patterns contain highly discriminative and forensically reliable identity information, supporting our hypothesis that gait analysis can expose deepfakes that preserve the source body's movement while substituting the target's face.

**Comprehensive Review of Deep Learning for Gait Analysis.** A recent survey [19] identifies various neural network approaches for gait-based anomaly detection in forensic applications, including GaitNet, ICDNet, and CNN-based architectures. The review emphasizes that gait analysis combined with deep learning enables identification of abnormal patterns from surveillance footage, but does not address deepfake detection specifically.

---

## D. Multimodal Deepfake Detection

Multimodal approaches combine multiple information streams for more robust forgery detection.

**Audio-Visual Fusion.** The Multi-Modal Multi-Sequence Bi-Modal Attention (MMMS-BA) framework [20] processes visual face sequences, lip sequences, and audio signals through separate bidirectional GRUs with cross-modal attention mechanisms. Evaluation on AV-DeepFake1M, FakeAVCeleb, LAV-DF, and TVIL datasets demonstrates a 3.47% increase in detection accuracy and 2.05% improvement in manipulation localization over prior methods.

**Human Perception Studies.** Research on human deepfake perception [21] reveals that visual appearance, vocal, and intuition cues co-occur for successful identifications. Audio cues are more important for confirming authenticity, while visual cues are more effective for detecting forgeries. These findings suggest that multimodal systems should weight modalities differently depending on the detection objective.

**Multimodal Gait Recognition.** MMGaitFormer [22] demonstrates that transformer-based fusion of skeleton and silhouette modalities improves gait recognition performance. While designed for identification rather than deepfake detection, the architectural principle — separate feature extraction followed by transformer-based fusion — could be adapted for multimodal deepfake detection incorporating gait, facial, and audio information.

---

## E. Research Gap and Our Contribution

The literature review reveals a significant research gap: **no existing comprehensive work explicitly combines gait-based skeletal analysis with deepfake detection.** The closest prior work [17] uses basic pose estimation with LSTM but lacks specialized gait features, biomechanical modeling, and the hybrid CNN+BiLSTM+Transformer architecture that enables both spatial feature extraction and multi-scale temporal analysis.

Our work addresses this gap with the following contributions:

1. **Novel problem formulation:** We are the first to frame deepfake detection as a gait verification problem — comparing the walking pattern in a video against an enrolled identity's gait signature, rather than searching for facial manipulation artifacts.

2. **Specialized gait feature engineering:** We extract 78-dimensional gait features per timestep using MediaPipe: 12 gait-relevant landmarks × 3D normalized coordinates (36 dims) + 6 biomechanical joint angles + 12-point velocities (36 dims). This feature representation captures both the spatial configuration and temporal dynamics of walking.

3. **Hybrid CNN+BiLSTM+Transformer architecture:** Our model combines a 1D CNN spatial encoder for frame-level gait features, a BiLSTM for short-range temporal dependencies and gait cycle modeling, and a Transformer encoder for long-range temporal attention — addressing the multi-scale temporal nature of gait patterns.

4. **Rigorous evaluation protocol:** We employ Leave-One-Subject-Out Cross-Validation (13 folds) to ensure no identity leakage, reporting AUC-ROC of 94.95% ± 2.81%, accuracy of 87.27% ± 3.76%, and EER of 13.19% ± 4.21% on 13 subjects with 1,056 augmented walking videos.

5. **Inherent advantages over face-based detection:** Our approach is (a) robust to face swap quality since it ignores facial features entirely, (b) privacy-preserving through skeleton-only analysis, and (c) grounded in forensically validated biometrics accepted as court evidence [18].

**TABLE III: COMPARATIVE POSITIONING OF OUR APPROACH**

| Method                     | Year     | Task                | Architecture         | Dataset    | Subjects | Key Metric     | Modality                   |
| -------------------------- | -------- | ------------------- | -------------------- | ---------- | -------- | -------------- | -------------------------- |
| GaitPT [3]                 | 2023     | Gait Recognition    | Transformer          | CASIA-B    | 124      | 82.6% Acc      | Skeleton                   |
| GaitFormer [4]             | 2023     | Gait Recognition    | ViT                  | CASIA-B    | 124      | 92.5% Acc      | Skeleton                   |
| AltFreezing [13]           | 2023     | Deepfake (Face)     | 3D ConvNet           | Celeb-DF   | —        | 96.7% AUC      | Face RGB                   |
| FTCN [12]                  | 2021     | Deepfake (Face)     | Temporal Conv+Trans  | FF++       | —        | 99.7% Acc      | Face RGB                   |
| Hybrid CNN-LSTM-Trans [15] | 2024     | Deepfake (Face)     | CNN+BiLSTM+Trans     | FF++       | —        | 98% AUC        | Face RGB                   |
| Pose+LSTM [17]             | 2025     | Deepfake (Motion)   | Pose+LSTM            | Custom     | —        | ~93% Acc       | Pose                       |
| **Ours**                   | **2026** | **Deepfake (Gait)** | **CNN+BiLSTM+Trans** | **Custom** | **13**   | **94.95% AUC** | **Skeleton (78-dim gait)** |

> **Note:** Direct numerical comparison across methods is limited since each operates on different datasets and tasks. However, our AUC of 94.95% is competitive with both gait recognition methods (which solve a different but related problem) and face-based deepfake detection methods (which use a fundamentally different modality). Our novelty lies in demonstrating that gait analysis is a viable and complementary approach to deepfake detection.

---

## References

[1] T. Teepe, A. Khan, J. Gilg, F. Herzog, S. Huhn, and G. Rigoll, "GaitGraph: Graph Convolutional Network for Skeleton-Based Gait Recognition," in _Proc. IEEE Int. Conf. Image Processing (ICIP)_, 2021. [Online]. Available: https://arxiv.org/abs/2101.11228

[2] B. Fu, F. Bouamer, N. Damer, and A. Kuijper, "GPGait: Generalized Pose-based Gait Recognition," in _Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)_, 2023. [Online]. Available: https://arxiv.org/abs/2303.05234

[3] M. Catruna, A. Cosma, and I. Radoi, "GaitPT: Skeletons Are All You Need for Gait Recognition," in _Proc. ACM Int. Conf. Multimedia (ACM MM)_, 2023. [Online]. Available: https://arxiv.org/abs/2308.10623

[4] A. Cosma and I. Radoi, "GaitFormer and DenseGait: Transformer-Based Gait Recognition with Dense Pretraining," 2023. [Online]. Available: https://arxiv.org/abs/2310.19418

[5] R. Sheng et al., "GaitTriViT and GaitVViT: Parallel Spatial-Temporal Vision Transformers for Gait Recognition," _PeerJ Computer Science_, vol. 11, e3061, 2025.

[6] S. Zhang et al., "Spatial Transformer Network Based Gait Recognition," _Expert Systems_, vol. 41, no. 1, e13244, 2023.

[7] "PoseMapGait: A Model-Based Gait Recognition Method with Pose Estimation and Graph Convolutional Network," Univ. Missouri-Kansas City. [Online]. Available: http://r.web.umkc.edu/rlyfv/papers/posemapgait.pdf

[8] "SPOSGait: Neural Architecture Search Based Gait Recognition," 2022. [Online]. Available: https://arxiv.org/abs/2205.02692

[9] M. Zhu et al., "Gait Recognition in the Wild: A Benchmark," in _Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)_, 2021.

[10] H. Zhao, W. Zhou, D. Chen, T. Wei, W. Zhang, and N. Yu, "Multi-Attentional Deepfake Detection," in _Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)_, 2021. [Online]. Available: https://arxiv.org/abs/2103.02406

[11] J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, and X. Yang, "End-to-End Reconstruction-Classification Learning for Face Forgery Detection," in _Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)_, 2022.

[12] Y. Zheng, J. Bao, D. Chen, M. Zeng, and F. Wen, "Exploring Temporal Coherence for More General Video Face Forgery Detection," in _Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)_, 2021. [Online]. Available: https://arxiv.org/abs/2108.06693

[13] Z. Wang, J. Bao, W. Zhou, W. Wang, and H. Li, "AltFreezing for More General Video Face Forgery Detection," in _Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)_, 2023.

[14] J. Le and D.-K. Woo, "FSBI: Frequency Enhanced Self-Blended Images for Deepfake Detection," 2024. [Online]. Available: https://arxiv.org/abs/2406.08625

[15] "Video Deepfake Detection Using a Hybrid CNN-LSTM-Transformer Architecture," _Zenodo_, 2024. [Online]. Available: https://zenodo.org/records/15862510

[16] "Deepfake-Eval-2024: A Comprehensive Multilingual Benchmark," 2024. [Online]. Available: https://arxiv.org/abs/2503.02857

[17] "DeepFake Detection: Exposing Deepfakes Through Smart Image Analysis," _Int. J. Creative Computing_, 2025. [Online]. Available: https://ijctjournal.org

[18] "Gait, Body Structure Biometrics Accepted as Court Evidence in EU Murder Case," _Biometric Update_, 2025. [Online]. Available: https://www.biometricupdate.com/202509

[19] "Gait Analysis Using Deep Learning: A Comprehensive Review for Forensic Applications," _PMC_, 2024. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC11622936/

[20] "Contextual Cross-Modal Attention for Audio-Visual Deepfake Detection (MMMS-BA)," _GitHub_, 2024. [Online]. Available: https://github.com/vcbsl/audio-visual-deepfake/

[21] "Human Perception of Deepfakes: Visual, Vocal, and Intuition Cues," 2025. [Online]. Available: https://arxiv.org/abs/2602.01284

[22] "MMGaitFormer: Multimodal Gait Recognition with Transformers," in _Proc. Semantic Scholar_. [Online]. Available: https://www.semanticscholar.org
