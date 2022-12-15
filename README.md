# Papers-VisualGrounding

- Detection
  - Single 
    - Referring Expression Comprehension 
    - grounding referring expression
  - multiple
    - Phrase localizing / grounding
    - Phrase extraction
- Segmentation
  - Referring image segmentation

Alternatively,

- fully- / weakly - Supervised
  - **Semi:** 
    - [WACV'21] Utilizing Every Image Object for Semi-supervised Phrase Grounding
    - [CRV'22] Semi-supervised Grounding Alignment for Multi-modal Feature Learning
    - [CVPR'22] Semi-supervised Video Paragraph Grounding with Contrastive Encoder
  - unsupervised: [CVPR'19] Phrase localization without paired training examples
- Image / video
- 2d / 3d



## 2022 papers

### CVRP22

> - https://cvpr2022.thecvf.com/accepted-papers

- Shifting More Attention to Visual Backbone: Query-modulated Refinement Networks for End-to-End Visual Grounding
- Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding
- Semi-supervised Video Paragraph Grounding with Contrastive Encoder
- Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning
- Weakly-Supervised Generation and Grounding of Visual Descriptions with Conditional Generative Models
- Multi-Modal Dynamic Graph Transformer for Visual Grounding



Referring image segmentation

- CRIS: CLIP-Driven Referring Image Segmentation
- Multi-Level Representation Learning with Semantic Alignment for Referring Video Object Segmentation
- Language-Bridged Spatial-Temporal Interaction for Referring Video Object Segmentation
- ReSTR: Convolution-free Referring Image Segmentation Using Transformers
- LAVT: Language-Aware Vision Transformer for Referring Image Segmentation
- Language as Queries for Referring Video Object Segmentation
- End-to-End Referring Video Object Segmentation with Multimodal Transformers



弱相关

- Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning
- TubeDETR: Spatio-Temporal Video Grounding with Transformers
- 3DJCG: A Unified Framework for Joint Dense Captioning and Visual Grounding on 3D Point Clouds
- 3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection
- Multi-View Transformer for 3D Visual Grounding
- Grounding Answers for Visual Questions Asked by Visually Impaired People
- Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs
- Invariant Grounding for Video Question Answering





### ECCV22

> https://eccv2022.ecva.net/program/accepted-papers/

- SeqTR: A Simple yet Universal Network for Visual Grounding
- SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding



弱相关

- Weakly Supervised Grounding for VQA in Vision-Language Transformers
- GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval
- Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding
- Asymmetric Relation Consistency Reasoning for Video Relation Grounding
- D3Net: A Unified Speaker-Listener Architecture for 3D Dense Captioning and Visual Grounding
- CYBORGS: Contrastively Bootstrapping Object Representations by Grounding in Segmentation
- SemAug: Semantically Meaningful Image Augmentations for Object Detection Through Language Grounding
- Grounding Visual Representations with Texts for Domain Generalization



### NIPS 22

> - https://nips.cc/Conferences/2022/Schedule?type=Poster

- Language Conditioned Spatial Relation Reasoning for 3D Object Grounding
- Embracing Consistency: A One-Stage Approach for Spatio-Temporal Video Grounding
- Look Around and Refer: 2D Synthetic Semantics Knowledge Distillation for 3D Visual Grounding 
- What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs



Refering Segmentation

- CoupAlign: Coupling Word-Pixel with Sentence-Mask Alignments for Referring Image Segmentation



### Others

- [ICLR](https://openreview.net/group?id=ICLR.cc/2022/Conference#spotlight-submissions)
  - Unsupervised Vision-Language Grammar Induction with Shared Structure Modeling
- AAAI23
  - DQ-DETR: Dual Query Detection Transformer for Phrase Extraction and Grounding
- ECCV workshop
  - YORO -- Lightweight End to End Visual Grounding 
- TPAMI: 
  - TransVG++: End-to-End Visual Grounding with Language Conditioned Vision Transformer




## 2021 papers

### CVPR21

> - https://openaccess.thecvf.com/CVPR2021

- [Relation-aware Instance Refinement for Weakly Supervised Visual Grounding](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Relation-aware_Instance_Refinement_for_Weakly_Supervised_Visual_Grounding_CVPR_2021_paper.html)
- [Co-Grounding Networks With Semantic Attention for Referring Expression Comprehension in Videos](https://openaccess.thecvf.com/content/CVPR2021/html/Song_Co-Grounding_Networks_With_Semantic_Attention_for_Referring_Expression_Comprehension_in_CVPR_2021_paper.html)
- [Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Improving_Weakly_Supervised_Visual_Grounding_by_Contrastive_Knowledge_Distillation_CVPR_2021_paper.html)
- [Iterative Shrinking for Referring Expression Grounding Using Deep Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Iterative_Shrinking_for_Referring_Expression_Grounding_Using_Deep_Reinforcement_Learning_CVPR_2021_paper.html)
- [Interventional Video Grounding With Dual Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Nan_Interventional_Video_Grounding_With_Dual_Contrastive_Learning_CVPR_2021_paper.html)
- [Refer-It-in-RGBD: A Bottom-Up Approach for 3D Visual Grounding in RGBD Images](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Refer-It-in-RGBD_A_Bottom-Up_Approach_for_3D_Visual_Grounding_in_RGBD_CVPR_2021_paper.html)
- [Embracing Uncertainty: Decoupling and De-Bias for Robust Temporal Grounding](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Embracing_Uncertainty_Decoupling_and_De-Bias_for_Robust_Temporal_Grounding_CVPR_2021_paper.html)
- [Cyclic Co-Learning of Sounding Object Visual Grounding and Sound Separation](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_Cyclic_Co-Learning_of_Sounding_Object_Visual_Grounding_and_Sound_Separation_CVPR_2021_paper.html)
- [Cascaded Prediction Network via Segment Tree for Temporal Video Grounding](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Cascaded_Prediction_Network_via_Segment_Tree_for_Temporal_Video_Grounding_CVPR_2021_paper.html)
- [Look Before You Leap: Learning Landmark Features for One-Stage Visual Grounding](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Look_Before_You_Leap_Learning_Landmark_Features_for_One-Stage_Visual_CVPR_2021_paper.html)

- [Scene-Intuitive Agent for Remote Embodied Visual Grounding](https://openaccess.thecvf.com/content/CVPR2021/html/Lin_Scene-Intuitive_Agent_for_Remote_Embodied_Visual_Grounding_CVPR_2021_paper.html)



### ICCV21

> - https://openaccess.thecvf.com/ICCV2021?day=all
- [MDETR - Modulated Detection for End-to-End Multi-Modal Understanding](https://openaccess.thecvf.com/content/ICCV2021/html/Kamath_MDETR_-_Modulated_Detection_for_End-to-End_Multi-Modal_Understanding_ICCV_2021_paper.html)
- [TransVG: End-to-End Visual Grounding With Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.html)
- [InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on Point Clouds Through Instance Multi-Level Contextual Referring](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_InstanceRefer_Cooperative_Holistic_Understanding_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.html)
- [SAT: 2D Semantics Assisted Training for 3D Visual Grounding](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_SAT_2D_Semantics_Assisted_Training_for_3D_Visual_Grounding_ICCV_2021_paper.html)
- [Free-Form Description Guided 3D Visual Graph Network for Object Grounding in Point Cloud](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Free-Form_Description_Guided_3D_Visual_Graph_Network_for_Object_Grounding_ICCV_2021_paper.html)
- [STVGBert: A Visual-Linguistic Transformer Based Framework for Spatio-Temporal Video Grounding](https://openaccess.thecvf.com/content/ICCV2021/html/Su_STVGBert_A_Visual-Linguistic_Transformer_Based_Framework_for_Spatio-Temporal_Video_Grounding_ICCV_2021_paper.html)
- [Detector-Free Weakly Supervised Grounding by Separation](https://openaccess.thecvf.com/content/ICCV2021/html/Arbelle_Detector-Free_Weakly_Supervised_Grounding_by_Separation_ICCV_2021_paper.html)
- [Support-Set Based Cross-Supervision for Video Grounding](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_Support-Set_Based_Cross-Supervision_for_Video_Grounding_ICCV_2021_paper.html)
- [Grounding Consistency: Distilling Spatial Common Sense for Precise Visual Relationship Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Diomataris_Grounding_Consistency_Distilling_Spatial_Common_Sense_for_Precise_Visual_Relationship_ICCV_2021_paper.html)
- [3DVG-Transformer: Relation Modeling for Visual Grounding on Point Clouds](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_3DVG-Transformer_Relation_Modeling_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.html)
- [Vision-Language Transformer and Query Generation for Referring Segmentation](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.html)







### NIPS21

> - https://papers.nips.cc/paper/2021

- [Explainable Semantic Space by Grounding Language to Vision with Cross-Modal Contrastive Learning](https://papers.nips.cc/paper/2021/hash/9a1335ef5ffebb0de9d089c4182e4868-Abstract.html) 
- [Referring Transformer: A One-step Approach to Multi-task Visual Grounding](https://papers.nips.cc/paper/2021/hash/a376802c0811f1b9088828288eb0d3f0-Abstract.html) 
- [End-to-end Multi-modal Video Temporal Grounding](https://papers.nips.cc/paper/2021/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html) 



## Previous

### Awesome series

- [Awesome Visual Grounding](https://github.com/TheShadow29/awesome-grounding)
- [Awesome Referring Expression Comprehension](https://github.com/daqingliu/awesome-rec)
- [Awesome-3D-Vision-and-Language](https://github.com/jianghaojun/Awesome-3D-Vision-and-Language)
- [paper with code](https://paperswithcode.com/task/visual-grounding/codeless?page=2&q=)



### CV-papers

- [Propagating Over Phrase Relations for One-Stage Visual Grounding](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_35)
- [Cross-Modal Relationship Inference for Grounding Referring Expressions](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Cross-Modal_Relationship_Inference_for_Grounding_Referring_Expressions_CVPR_2019_paper.html)
- Structured Attention Network for Referring Image Segmentation



- MAttNet: Modular Attention Network for Referring Expression Comprehension



## Arxiv Papers



# Gossip

### Dataset

- Referring expression comprehension / Grounding Referring Expressions

  > - based on: COCO
  > - simple object: easier than phrase grounding

  - refCOCO
  - refCOCO+
  - refCOCOg
  - refer to https://github.com/GangyiTian/visual-grounding-dataset

- Phrase grounding

  > - more chanllenging due to multiple phrase/objects

  - Flickr30k Entities
  - ReferItGame

- Referring Image Segmentation
  - UNC / UNC+
  - G-Ref
  - ReferIt


### PhDs

- [Arka Sadhu](https://theshadow29.github.io/)
- [Zhengyuan Yang](https://zyang-ur.github.io/) (one-stage)
  - Thesis: https://www.proquest.com/docview/2572612109
- [Jiajun Deng](https://scholar.google.com/citations?user=FAAHjxsAAAAJ&hl=zh-CN) (Weakly)
- [Daqing Liu](https://scholar.google.com/citations?hl=en&user=TbBfOVEAAAAJ&view_op=list_works&sortby=pubdate) (Struture-language)
- [Haojun Jiang](https://github.com/jianghaojun) (label-efficient)
- [Liwei Wang](https://lwwangcse.github.io/) (3D grounding)
- [Sibei yang](https://sibeiyang.github.io/) (refering expression)
- [Licheng Yu](https://lichengunc.github.io/) (before:refering expression, now:pretrain,generations)



