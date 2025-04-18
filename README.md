# WOLF
## 🐾 Wildlife Observation through Latent Feature evaluation 📸

## Overview

This project implements and compares several deep learning models for classifying images of 21 different wild animals. It utilizes transfer learning techniques on pre-trained architectures to achieve efficient training and robust performance. The project follows a modular structure covering dataset preparation, model training, and evaluation.

## Goal 🎯

To accurately classify images into one of the following 21 wild animal categories:
`BEAR` 🐻, `BISON`🐃, `CHEETAH`🐆, `ELEPHANT`🐘, `FOX`🦊, `GAZELLE`, `GIRAFFE`🦒, `GORILLA`🦍, `HIPPO`, `HORRSE`🐎, `HYENA`, `KOALA`🐨, `LEOPARD`, `LION`🦁, `MEERKAT`, `PIG`🐖, `PORCUPINE`🦔, `RHINO`🦏, `TIGER`🐅, `WOLF`🐺, `ZEBRA`🦓.

## Dataset 💾

The project uses the "Zoo 2000 - Wild Animals" dataset available on Kaggle:
`/kaggle/input/zoo-2000/Wild_Animals/` 🔗

The dataset is pre-split into `Train`, `Validation`, and `Test` directories, with subdirectories for each of the 21 classes.

## Methodology 🛠️

*   **Models:** Explores three pre-trained CNN architectures:
    *   MobileNetV2 📱
    *   ResNet50 🧱
    *   ConvNeXt Base ✨
*   **Transfer Learning:** Utilizes weights pre-trained on ImageNet. The base models are initially frozen, and only a custom classification head is trained. 🧠
*   **Data Pipeline:** Employs `tf.data` pipelines for efficient data loading and preprocessing, including disk caching. ⚡
*   **Preprocessing:** Applies model-specific preprocessing functions (`preprocess_input`) and standard resizing (224x224). 🖼️
*   **Data Augmentation:** Uses standard augmentation techniques (flips, rotations, zooms) on the training set to improve generalization. 🎲
*   **Class Imbalance Handling:** Calculates and uses class weights during training to mitigate the effects of imbalanced class distributions in the dataset. ⚖️
*   **Training:** Uses Adam optimizer, categorical cross-entropy loss, and callbacks (`EarlyStopping`, `ModelCheckpoint`). 📈
*   **Evaluation:** Assesses model performance on the unseen test set using:
    *   Overall Accuracy & Top-5 Accuracy ✅
    *   Classification Report (Precision, Recall, F1-Score per class) 📊
    *   Confusion Matrix (Raw counts and Normalized) 🔢
*   **Visualization:** Includes code to visualize intermediate layer activations to understand feature extraction. 👀

## Project Structure 📂

The project is organized into modules, typically implemented within Kaggle notebooks:

1.  **Module 1: Dataset Preparation & Exploration** 🌍
    *   Loading and organizing data splits.
    *   Visualizing samples and class distributions.
    *   Calculating class weights.
    *   Defining baseline preprocessing and augmentation strategies.
    *   Outputs saved to `module_1_outputs/` (e.g., class weights JSON, plots).
2.  **Module 2: Model Implementation & Training** 🤖
    *   Loading pre-trained models (MobileNetV2, ResNet50, ConvNeXt Base).
    *   Freezing base layers and adding custom classification heads.
    *   Implementing `tf.data` pipelines with model-specific preprocessing and augmentation.
    *   Training models using class weights and callbacks.
    *   Visualizing intermediate layer activations.
    *   Outputs saved to `module_2_outputs/` (e.g., `.keras` models, `.pkl` history files).
3.  **Module 3: Evaluation & Comparison** 🤔
    *   Loading saved models from Module 2 outputs.
    *   Preparing the test dataset generator.
    *   Evaluating models on the test set.
    *   Generating classification reports and confusion matrices.
    *   Comparing the performance of the different architectures.
    *   Outputs saved to `module_3_outputs/` (e.g., reports, confusion matrix plots).

*(Note: In Kaggle, outputs from one notebook version can be used as inputs for the next module's notebook).*

## Setup ⚙️

*   **Environment:** Primarily developed using Kaggle Notebooks with GPU acceleration. 💻 + 🔥
*   **Key Libraries:**
    *   TensorFlow / Keras (`tensorflow`)
    *   Scikit-learn (`sklearn`)
    *   NumPy (`numpy`)
    *   Matplotlib (`matplotlib`)
    *   Seaborn (`seaborn`)
    *   Pillow (for image loading via Keras utils)
    *   Pickle (`pickle`)
    *   JSON (`json`)

*(See individual notebooks for specific imports)*

## Usage ▶️

1.  Ensure the "Zoo 2000 - Wild Animals" dataset is available (e.g., attached to your Kaggle notebook).
2.  Run the notebook corresponding to Module 1 to prepare data and calculate weights.
3.  Run the notebook corresponding to Module 2 (same as Module 1) to train the desired models (MobileNetV2, ResNet50, ConvNeXt Base). Ensure the outputs (models, history) are saved using "Save Version".
4.  Run the notebook corresponding to Module 3. Add the output from the Module 2 notebook version as an input source. The notebook will load the saved models and evaluate them on the test set.

## Results Summary 🏆

The trained models were evaluated on the test set. Key performance metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) for each model can be found in the outputs generated by the Module 3 notebook (specifically in `module_3_outputs/`).

## Results Summary 🏆

The performance of the three models, trained via transfer learning with a frozen base, was evaluated on the test set. Key metrics are summarized below:

| Model           | Test Accuracy | Weighted Avg F1-Score | Macro Avg F1-Score | Link to Report                             |
| :-------------- | :-----------: | :-------------------: | :----------------: | :----------------------------------------- |
| **ResNet50**    |   **94.5%**   |      **0.945**        |       0.944        | `module_3_outputs/resnet50_classification_report.txt`    |
| MobileNetV2     |     94.1%     |         0.941         |       0.939        | `module_3_outputs/mobilenetv2_classification_report.txt` |
| ConvNeXt Base   |     83.8%     |         0.839         |       0.846        | `module_3_outputs/convnext_base_classification_report.txt`|

**Conclusion:** 
*   🥇 **ResNet50:** Emerged as the top performer in this initial phase, achieving an overall **accuracy of 94.5%**. It also scored highest on the **weighted average F1-score (0.945)**, indicating strong and balanced performance across classes relative to their size in the test set.
*   🥈 **MobileNetV2:** Showed highly competitive results, nearly matching ResNet50 with **94.1% accuracy** and a **weighted average F1-score of 0.941**. This highlights its effectiveness even as a more lightweight architecture.
*   🥉 **ConvNeXt Base:** While a more complex model, it yielded a lower **accuracy of 83.8%** and a **weighted average F1-score of 0.839** when used solely as a frozen feature extractor. This suggests that its powerful features may require adaptation through fine-tuning to reach their full potential on this specific dataset.

**Note:** All models were trained using identical data pipelines, augmentation strategies, and class weighting. The performance differences primarily reflect the suitability of the frozen pre-trained features for this task.

Full classification reports and confusion matrices are available in the `module_3_outputs/` directory.

## Future Work 🚀

*   **Fine-tuning:** Unfreeze later layers of the base models and retrain with a lower learning rate.
*   **Hyperparameter Tuning:** Experiment with different learning rates, optimizers, or classification head architectures.
*   **Advanced Augmentation:** Explore more sophisticated augmentation techniques.
*   **Multi-GPU Training:** Implement strategies like `MirroredStrategy` for faster training if multiple GPUs are available.
*   **Ensemble Methods:** Combine predictions from multiple models.
