# Lab 2 – Deep Learning avec PyTorch (GPU)

## 1. Objectif
Prendre en main PyTorch pour construire des architectures de vision par ordinateur (CNN, Faster R-CNN, modèles pré-entraînés, Vision Transformer) et les comparer sur MNIST, en mode GPU (Tesla T4, Google Colab).

## 2. Contenu du dépôt
| Fichier | Description |
|---|---|
| `Lab2_DeepLearning.ipynb` | Notebook complet exécuté (code + sorties) |
| `README.md` | Ce rapport |

## 3. Données
MNIST : 60 000 images d'entraînement, 10 000 de test (28×28, 10 classes), chargé via `torchvision.datasets.MNIST` (identique au dataset Kaggle `hojjatk/mnist-dataset`).

## 4. Modèles et hyper-paramètres

### 4.1 CNN
- 2 blocs `Conv(3×3, padding 1, stride 1) → BatchNorm → ReLU → MaxPool(2×2)` (32 puis 64 filtres), puis `Dropout(0.5) → FC(3136→128) → ReLU → Dropout(0.3) → FC(128→10)`.
- Adam (lr 1e-3, weight decay 1e-4), StepLR (γ = 0.5 toutes les 2 epochs), 5 epochs, batch 128, CrossEntropy. Données complètes.
- 421 834 paramètres.

### 4.2 Faster R-CNN
- MNIST converti en problème de détection : boîte englobante calculée à partir des pixels du chiffre, classe = chiffre + 1 (0 = fond).
- `fasterrcnn_mobilenet_v3_large_320_fpn` (backbone ImageNet), images 128×128, SGD (lr 0.01, momentum 0.9, weight decay 5e-4), 3 epochs, sous-ensemble de 10 000 images d'entraînement / 2 000 de test.
- Classification = label de la détection ayant le meilleur score.
- 18,98 M de paramètres.

### 4.3 Fine-tuning de VGG16 et AlexNet
- Poids ImageNet, images redimensionnées à 64×64 sur 3 canaux, normalisation ImageNet.
- Backbone convolutif gelé, dernière couche FC remplacée (4096 → 10), classifieur ré-entraîné (Adam, lr 1e-4), 2 epochs, 20 000 images d'entraînement / 5 000 de test.

### 4.4 Vision Transformer (from scratch)
- Patchs 7×7 (16 patchs) + token [CLS] + encodage positionnel appris, 4 blocs Transformer (4 têtes, dimension 64, MLP ×4, dropout 0.1).
- AdamW (lr 1e-3, weight decay 0.05) + CosineAnnealing, 10 epochs, données complètes.
- 205 066 paramètres.

## 5. Résultats (mesurés sur Tesla T4)

Les courbes, matrices de confusion et graphiques de comparaison sont disponibles dans le notebook `Lab2_DeepLearning.ipynb`.

| Modèle | Accuracy | F1 (macro) | Loss (test) | Temps d'entraînement (s) | Paramètres |
|---|---|---|---|---|---|
| **CNN** | **0.9907** | **0.9907** | **0.0253** | 80.6 | 0,42 M |
| ViT (from scratch) | 0.9840 | 0.9839 | 0.0522 | 193.0 | 0,21 M |
| VGG16 (fine-tuned) | 0.9620 | 0.9616 | 0.1167 | 49.7 | 134,3 M (119,6 M entraînables) |
| AlexNet (fine-tuned) | 0.9372 | 0.9360 | 0.1826 | 26.7 | 57,0 M (54,6 M entraînables) |
| Faster R-CNN | échec (loss = NaN) | – | NaN | 493.1 | 18,98 M |

> **Attention à la lecture des temps** : ils ne portent pas sur les mêmes volumes de données. CNN : 60 000 images × 5 epochs ; ViT : 60 000 × 10 epochs ; VGG16/AlexNet : 20 000 × 2 epochs ; Faster R-CNN : 10 000 × 3 epochs.

### 5.1 CNN
Accuracy test de 98,32 % dès la 1ʳᵉ epoch, 99,07 % à la 5ᵉ ; la loss de test baisse régulièrement (0.0486 → 0.0253) sans signe de sur-apprentissage.

### 5.2 Faster R-CNN : entraînement non convergent
La loss est devenue `NaN` dès la 1ʳᵉ epoch, le modèle ne produit donc aucune détection (accuracy = 0 %, aucune boîte n'est affichée dans le notebook). Il s'agit d'une **divergence numérique de l'entraînement**, pas d'une limite du modèle : la combinaison SGD avec un learning rate de 0.01 sur des têtes initialisées aléatoirement est probablement trop agressive. Les résultats de Faster R-CNN ne sont donc **pas exploitables** dans cette version ; une nouvelle exécution avec un learning rate plus faible (≈ 0.002), un warm-up et un gradient clipping est nécessaire pour obtenir une vraie comparaison avec le CNN.

### 5.3 VGG16 et AlexNet (fine-tuning)
Après 2 epochs, VGG16 atteint 96,20 % et AlexNet 93,72 %. Les deux restent en dessous du CNN simple.

### 5.4 Vision Transformer
L'accuracy passe de 92,45 % (epoch 1) à 98,40 % (epoch 10). La courbe est encore ascendante, mais la loss de test se stabilise vers 0.052.

## 6. Analyse et conclusion
- **CNN** : meilleur modèle (99,07 %) avec un coût très faible. Les convolutions (localité, partage de poids, invariance à la translation) sont parfaitement adaptées à des chiffres manuscrits centrés.
- **ViT** : 98,40 %, soit environ 0,7 point de moins que le CNN avec deux fois plus d'epochs et un temps d'entraînement environ 2,4 fois plus long. Sans biais inductif propre aux images, il doit apprendre la structure spatiale à partir des données ; il reste néanmoins très performant avec seulement 0,2 M de paramètres.
- **VGG16 / AlexNet** : moins bons que le CNN (96,2 % et 93,7 %) malgré 134 M et 57 M de paramètres. Explications probables : backbone gelé dont les caractéristiques ImageNet (images naturelles en couleur) sont peu adaptées à des chiffres en niveaux de gris upscalés en 64×64, seulement 2 epochs, et un tiers des données d'entraînement. VGG16 (plus profond) fait mieux qu'AlexNet. Le transfert offre un entraînement court par epoch, mais un coût mémoire/calcul très supérieur pour une précision plus faible sur cette tâche simple.
- **Faster R-CNN** : l'entraînement a divergé (NaN), aucune conclusion de performance ne peut être tirée. Par construction, c'est un détecteur (RPN + tête de classification + régression de boîtes), nettement plus lourd qu'un classifieur pour une tâche d'un seul chiffre par image ; sa loss somme plusieurs termes et n'est pas comparable à une cross-entropy.
- **Conclusion** : sur MNIST, un petit CNN offre le meilleur compromis précision / coût. Le ViT s'en approche mais est plus coûteux ; le fine-tuning de VGG16/AlexNet n'est pas rentable ici. Les ViT deviennent avantageux avec de grands jeux de données ou un pré-entraînement massif.

## 7. Synthèse personnelle
_À rédiger avec vos mots. Points possibles : manipulation de PyTorch (Dataset/DataLoader, boucle d'entraînement, GPU), conception d'un CNN et choix des hyper-paramètres, adaptation d'un détecteur à la classification et difficulté de stabiliser son entraînement (divergence NaN), fine-tuning de réseaux pré-entraînés (gel du backbone, nouvelle tête), implémentation de l'attention multi-têtes et d'un ViT, comparaison rigoureuse de modèles (Accuracy, F1, loss, temps) en tenant compte des volumes de données différents._

## 8. Exécution
1. Ouvrir `Lab2_DeepLearning.ipynb` dans Google Colab ou Kaggle.
2. Activer le GPU (*Exécution → Modifier le type d'exécution → GPU*).
3. *Tout exécuter* (≈ 15–25 min sur T4). Tailles de sous-ensembles et nombre d'epochs réglables dans le dictionnaire `CFG`.
