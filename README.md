DCGAN vs WGAN-GP Tackling Mode Collapse in GANs

Generative AI (AI4009) | Assignment No. 3 | Spring 2026 | FAST NUCES
Authors: Bushra Abad & Tayyaba Imtiyaz


Overview
This project implements and compares two GAN architectures from scratch in PyTorch to address the problem of mode collapse in generative models:

DCGAN — Deep Convolutional GAN (baseline)
WGAN-GP — Wasserstein GAN with Gradient Penalty (improved)

Both models are trained on the Anime Faces dataset (43,102 images) to generate 64×64 anime character faces. The goal is to demonstrate how replacing Binary Cross Entropy loss with Wasserstein loss + Gradient Penalty leads to more stable training and greater diversity in generated outputs.

The Problem: Mode Collapse
Mode collapse happens when the Generator finds a small set of outputs that consistently fool the Discriminator and stops exploring the rest of the data distribution. The result: a model that generates many variations of the same face instead of diverse, creative outputs.
DCGAN is prone to this because BCE loss can saturate, killing the gradient signal when the Discriminator becomes too confident. WGAN-GP fixes this at the mathematical level by using Wasserstein distance  a smoother, non-saturating measure of how far apart two distributions are.

Architecture
Generator (Shared by Both Models)
Noise z (100, 1, 1)
    → ConvTranspose2d → 512 x 4 x 4    (BatchNorm + ReLU)
    → ConvTranspose2d → 256 x 8 x 8    (BatchNorm + ReLU)
    → ConvTranspose2d → 128 x 16 x 16  (BatchNorm + ReLU)
    → ConvTranspose2d → 64  x 32 x 32  (BatchNorm + ReLU)
    → ConvTranspose2d → 3   x 64 x 64  (Tanh)
DCGAN Discriminator
Image (3, 64, 64)
    → Conv2d → 64  x 32 x 32  (LeakyReLU 0.2)
    → Conv2d → 128 x 16 x 16  (BatchNorm + LeakyReLU 0.2)
    → Conv2d → 256 x 8  x 8   (BatchNorm + LeakyReLU 0.2)
    → Conv2d → 512 x 4  x 4   (BatchNorm + LeakyReLU 0.2)
    → Conv2d → 1               (Sigmoid → probability)
WGAN-GP Critic
Image (3, 64, 64)
    → Conv2d → 64  x 32 x 32  (LeakyReLU 0.2)
    → Conv2d → 128 x 16 x 16  (InstanceNorm + LeakyReLU 0.2)
    → Conv2d → 256 x 8  x 8   (InstanceNorm + LeakyReLU 0.2)
    → Conv2d → 512 x 4  x 4   (InstanceNorm + LeakyReLU 0.2)
    → Conv2d → 1               (No Sigmoid → unbounded score)

Why InstanceNorm in the Critic? BatchNorm conflicts with gradient penalty computation. InstanceNorm normalizes each sample independently, avoiding this instability.

Model Parameters
ComponentParametersGenerator (shared)3,576,704DCGAN Discriminator2,765,568WGAN-GP Critic2,765,568

Key Differences: DCGAN vs WGAN-GP
DCGANWGAN-GPLoss FunctionBinary Cross EntropyWasserstein LossDiscriminator OutputProbability (Sigmoid)Unbounded Score (no Sigmoid)NormalizationBatchNormInstanceNorm (in Critic)Constraint MethodNoneGradient Penalty (λ=10)Critic Updates per G Update15Mode Collapse ResistanceLowHigh

Dataset
Anime Faces — 43,102 anime character face images, 64×64

Platform: Kaggle
Dataset: soumikrakshit/anime-faces
Accelerator: Kaggle T4 × 2 (Dual GPU)


Training Details
HyperparameterValueImage Size64 × 64Noise Dimension (z)100Batch Size64Learning Rate0.0002Adam Betas(0.5, 0.999)DCGAN Epochs15WGAN-GP Epochs15Critic Iterations5Gradient Penalty λ10PrecisionMixed (torch.cuda.amp)

Loss Functions
DCGAN — Binary Cross Entropy
pythoncriterion = nn.BCEWithLogitsLoss()

# Discriminator loss
d_loss = criterion(D(real), ones) + criterion(D(fake), zeros)

# Generator loss
g_loss = criterion(D(G(z)), ones)
WGAN-GP — Wasserstein Loss + Gradient Penalty
python# Critic loss (minimize negative Wasserstein distance + GP)
c_loss = -(critic(real).mean() - critic(fake).mean()) + lambda_gp * gradient_penalty(...)

# Generator loss (maximize critic score on fakes)
g_loss = -critic(G(z)).mean()
Gradient Penalty
pythondef gradient_penalty(critic, real, fake):
    alpha        = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score        = critic(interpolated)
    grad         = torch.autograd.grad(score, interpolated,
                       grad_outputs=torch.ones_like(score),
                       create_graph=True, retain_graph=True)[0]
    gp = ((grad.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
    return gp

Project Structure
├── 22f_3324_22f_3863_AI_ASS03_GAN.ipynb   # Main notebook
├── checkpoints/
│   ├── dcgan_G_final.pt
│   ├── dcgan_D_final.pt
│   ├── wgan_G_final.pt
│   ├── wgan_C_final.pt
│   └── losses.json
└── README.md
Notebook Sections

Environment Setup — installs, imports, hyperparameters
Data Preparation — Anime Faces download, transforms, DataLoader
Model Architecture — Generator, DCGAN Discriminator, WGAN-GP Critic, Gradient Penalty
Training — DCGAN loop, WGAN-GP loop (with 5:1 critic ratio)
Loss Plots — Generator & Discriminator/Critic loss vs epochs
Visualization — 16-sample grids per model + side-by-side comparison
Save Checkpoints — model weights + loss history as JSON
Gradio App — interactive demo with both models


Gradio App
An interactive Gradio app lets you:

Choose the number of images to generate (4–16)
Set a random seed for reproducibility
Generate and compare outputs from both DCGAN and WGAN-GP side by side in real time


How to Run
On Kaggle (Recommended)

Create a new Kaggle notebook
Add the Anime Faces dataset
Enable GPU T4 × 2 accelerator
Upload and run 22f_3324_22f_3863_AI_ASS03_GAN.ipynb

Dependencies
bashpip install torch torchvision gradio

Important Implementation Notes

No Sigmoid in WGAN-GP Critic — the output is an unbounded real score, not a probability
InstanceNorm replaces BatchNorm in Critic — required for stable gradient penalty computation
5 Critic updates per 1 Generator update — keeps the Wasserstein estimate accurate
Adam betas (0.5, 0.999) — lower first-moment momentum reduces oscillation in adversarial training
.detach() when training Discriminator — stops gradients from flowing into Generator during D step
Fixed noise vector for visualization — same z input used throughout training to track progress consistently


Note on Results
Both models were trained for 15 epochs under Kaggle's free GPU memory constraints (batch size 64, mixed precision). Results are solid proof-of-concept outputs demonstrating the stability difference between the two approaches. With more epochs and compute, visual quality would improve significantly — especially for WGAN-GP, which tends to keep improving longer without collapsing.

References

Radford, A. et al. (2015). Unsupervised Representation Learning with Deep Convolutional GANs
Arjovsky, M. et al. (2017). Wasserstein GAN
Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs


Authors
NameRoll NoBushra Abad22F-3863Tayyaba Imtiyaz22F-3324
Course: Generative AI (AI4009) | Semester: Spring 2026 | Institution: FAST NUCES
