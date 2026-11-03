# Lab 2 – Deep Learning avec PyTorch (GPU)

**Université Abdelmalek Essaadi – FST Tanger – Département Génie Informatique – Cycle LSI**
**Module :** Deep Learning – Pr. ELAACHAK Lotfi
**Étudiant :** _Votre nom_

## 1. Objectif
Prendre en main PyTorch pour construire des architectures de vision par ordinateur (CNN, Faster R-CNN, modèles pré-entraînés, Vision Transformer) et les comparer sur MNIST, en mode GPU.

## 2. Contenu du dépôt
| Fichier | Description |
|---|---|
| `Lab2_DeepLearning.ipynb` | Notebook complet (Google Colab / Kaggle, GPU) |
| `resultats_lab2.csv` | Tableau final des métriques (généré par le notebook) |
| `README.md` | Ce rapport |

## 3. Données
MNIST : 60 000 images d'entraînement, 10 000 de test (28×28, 10 classes), chargé via `torchvision.datasets.MNIST` (identique au dataset Kaggle `hojjatk/mnist-dataset`).

## 4. Modèles

### 4.1 CNN
2 blocs `Conv(3×3, padding 1, stride 1) → BatchNorm → ReLU → MaxPool(2×2)` (32 puis 64 filtres), puis `Dropout → FC(3136→128) → ReLU → Dropout → FC(128→10)`.
Adam (lr 1e-3, weight decay 1e-4), StepLR, 5 epochs, CrossEntropy.

### 4.2 Faster R-CNN
MNIST converti en détection : boîte englobante calculée à partir des pixels du chiffre, classe = chiffre + 1. Modèle `fasterrcnn_mobilenet_v3_large_320_fpn` (backbone ImageNet), images 128×128, SGD (lr 0.01, momentum 0.9), entraînement sur un sous-ensemble (10 000 train / 2 000 test). Classification = label de la détection au meilleur score.

### 4.3 Fine-tuning VGG16 / AlexNet
Poids ImageNet, images redimensionnées en 64×64 sur 3 canaux, backbone convolutif gelé, dernière couche FC remplacée (10 classes), classifieur ré-entraîné (Adam, lr 1e-4, 2 epochs, 20 000 train / 5 000 test).

### 4.4 Vision Transformer (from scratch)
Patchs 7×7 (16 patchs) + token [CLS] + encodage positionnel appris, 4 blocs Transformer (4 têtes, dim 64, MLP ×4), AdamW (lr 1e-3, wd 0.05) + cosine, 10 epochs.

## 5. Résultats
> ⚠️ Remplacez ce tableau par les valeurs obtenues après exécution du notebook (`resultats_lab2.csv`).

| Modèle | Accuracy | F1 (macro) | Loss (test) | Temps d'entraînement (s) | Paramètres |
|---|---|---|---|---|---|
| CNN | _…_ | _…_ | _…_ | _…_ | ~0,42 M |
| Faster R-CNN | _…_ | _…_ | _…_ | _…_ | _…_ |
| AlexNet (fine-tuned) | _…_ | _…_ | _…_ | _…_ | ~57 M |
| VGG16 (fine-tuned) | _…_ | _…_ | _…_ | _…_ | ~134 M |
| ViT (from scratch) | _…_ | _…_ | _…_ | _…_ | ~0,2 M |

_Ajoutez ici les courbes loss/accuracy et les matrices de confusion exportées depuis le notebook._

## 6. Analyse et conclusion
- **CNN** : excellent compromis précision / coût, car les convolutions exploitent la structure locale des images.
- **Faster R-CNN** : conçu pour la détection, il est inutilement lourd et lent pour classer un chiffre unique ; sa loss (classification + boîtes + RPN) n'est pas comparable à une cross-entropy.
- **VGG16 / AlexNet** : convergence rapide grâce au transfert, mais très coûteux en paramètres et en calcul pour un gain marginal sur MNIST.
- **ViT** : sans biais inductif, il a besoin de plus de données/epochs ; performances correctes mais généralement inférieures au CNN sur MNIST à budget égal. Il devient avantageux sur de grands jeux de données ou avec pré-entraînement.
- **Conclusion** : _à adapter à vos résultats chiffrés_.

## 7. Synthèse personnelle
_Ce que j'ai appris durant ce lab : manipulation de PyTorch et du GPU, conception d'un CNN et choix des hyper-paramètres, adaptation de Faster R-CNN à la classification, fine-tuning de modèles pré-entraînés, implémentation d'un ViT (attention multi-têtes), comparaison rigoureuse de modèles. (À rédiger avec vos mots.)_

## 8. Exécution
1. Ouvrir `Lab2_DeepLearning.ipynb` dans Google Colab ou Kaggle.
2. Activer le GPU (*Exécution → Modifier le type d'exécution → GPU*).
3. *Tout exécuter* (≈ 15–25 min sur T4). Les tailles de sous-ensembles et le nombre d'epochs se règlent dans le dictionnaire `CFG`.
