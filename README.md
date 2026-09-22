# MetaClip
MetaCLIP-CMR

MetaCLIP-CMR is an image-text pre-training project for cardiac magnetic resonance imaging (CMR). It generates text descriptions from structured metadata embedded in image file paths, trains image and text encoders through contrastive learning, and transfers the image encoder to downstream classification, segmentation, and disease recognition tasks.

Project Workflow

The workflow consists of data indexing, image preprocessing, metadata-based text generation, multimodal pre-training, downstream training, and evaluation. An image reconstruction branch provides a separate pre-training baseline.

1. Organize and Index the Raw Data

Scan the raw CMR dataset for .mat files and record their paths. Parse modality, view, centre, scanner vendor, field strength, and patient identifiers to construct a data index.

Relevant scripts:

build_data_txt.py: Collect paths to raw .mat files.

build_data_index.py: Parse path metadata and generate a CSV index.

The project covers ten modalities: Cine, LGE, Mapping, Flow2d, Aorta, Perfusion, T1w, T2w, Tagging, and T1rho.

2. Convert Images and Prepare Training Data

Convert the original images into two-dimensional .npy files for individual slices and frames. Store image paths, source sequence paths, slice indices, frame indices, and metadata alongside them. Existing Cine image caches can be incorporated through their existing indices.

The training scripts read CSV or Parquet indices, group images by source sequence, and perform slice and frame sampling, intensity normalization, resizing, and training-time augmentation.

Relevant scripts:

convert_mat_to_npy.py: Convert images and build a unified NPY index.

cmr_multimodal_dataset.py: Provide components for multimodal sequence loading, sampling, and text generation.

train_cmr_multimodal.py: Include the dataset implementation used for multimodal pre-training.

3. Generate Text Descriptions from Metadata

Extract modality, view, vendor, scanner model, and field strength from file paths and sequence information. Insert these attributes into predefined templates to create text descriptions paired with the images.

For example, a description may identify a short-axis cine cardiac MRI sequence acquired on a Siemens 3.0T scanner. Training randomly selects among templates describing the same metadata, while evaluation uses a fixed template.

These descriptions provide acquisition and image-type information without requiring a manually written clinical report for each sequence.

4. Perform Image-Text Contrastive Pre-training

train_cmr_multimodal.py jointly trains on images and their corresponding metadata descriptions:

Extract image representations using a ResNet encoder adapted for single-channel input, with ResNet50 as the default backbone.

Extract text representations using DistilBERT.

Map both representations into a shared 256-dimensional space using projection heads and apply L2 normalization.

Compute image-text similarities within each batch and optimize a bidirectional contrastive loss.

Assign soft targets to pairs sharing a modality or view to account for their semantic similarity.

Training initially freezes the backbone networks, then unfreezes them for joint optimization. Model checkpoints are saved for downstream use.

5. Train the Image Reconstruction Baseline

train_mim.py provides an image-only masked reconstruction pre-training branch:

Select a two-dimensional image from the sampled images.

Randomly mask a subset of image blocks.

Reconstruct the image using a ResNet50 encoder and a reconstruction decoder.

Compute reconstruction loss at masked pixel locations.

Save the encoder weights for downstream transfer experiments.

This branch serves as a comparison with image-text pre-training.

6. Transfer the Image Encoder to Downstream Tasks

After multimodal pre-training, downstream models use the image encoder for feature extraction without requiring text input. A classification head or segmentation decoder is attached for each task. Training includes a stage with the encoder frozen, followed by fine-tuning with the encoder unfrozen.

Task

Data

Script

Workflow

Modality classification

Multimodal CMR sequences

train_classification.py

Classify seven selected imaging modalities

Cine view classification

Cine sequences

train_classification.py

Distinguish 2ch, 3ch, 4ch, and SAX views

Cine SAX segmentation

ACDC and M&Ms

train_sax_seg.py

Segment the right ventricle, myocardium, and left ventricle using a ResNet50 U-Net

LGE segmentation

EMIDEC

train_emidec.py

Segment the left ventricular cavity, myocardium, and scar regions

ACDC disease classification

ACDC

train_acdc_disease.py

Aggregate features from multiple end-diastolic (ED) and end-systolic (ES) slices for five-class disease classification

LGE disease recognition

EMIDEC

train_emidec.py

Classify cases as normal or myocardial infarction

Segmentation data are prepared using preprocess_sax_seg.py and preprocess_emidec.py, which convert NIfTI images and annotations into NPY files and accompanying indices.

7. Evaluate Models and Export Results

Pre-training evaluation computes image-text retrieval metrics and modality matching accuracy. Downstream classification produces accuracy, classification reports, and confusion matrices. Segmentation evaluation aggregates predictions by case and computes Dice scores for each structure. EMIDEC disease recognition aggregates prediction probabilities by case and reports accuracy, AUC, sensitivity, and specificity.

Saved downstream checkpoints can be evaluated and their predictions exported using:

eval_cls_checkpoint.py: Export each classification sample's ground-truth label, predicted label, and correctness indicator.

eval_seg_checkpoint.py: Export per-structure and mean Dice scores for each ACDC or M&Ms case.

The exported results support comparisons across initialization and pre-training strategies, further statistical analysis, and visualization.
