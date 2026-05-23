# [TR]
# Otonom Savaş Pilotu AI (Autonomous BVR Fighter AI) ✈️🧠

Bu proje, F-16 savaş uçaklarının uçuş dinamiklerini **JSBSim** açık kaynaklı uçuş fizik motoru kullanarak simüle eden ve **Soft Actor-Critic (SAC)** algoritması ile otonom bir savaş pilotu yetiştirmeyi amaçlayan bir yapay zeka oluşturma çalışmasıdır.

## 🚀 Proje Aşamaları

Proje, ajanın sıfır bilgiden Görüş Ötesi Çatışma (BVR) yeteneğine ulaşması için çok aşamalı bir "Müfredatlı Öğrenme" (Curriculum Learning) konseptiyle tasarlanmıştır.

* **[X] Faz 1: Aerodinamik Hayatta Kalma (Trim & Survival)** - *Tamamlandı*
* **[X] Faz 2: Seyrüsefer ve Komut Takibi (Navigation & Command Following)** - *Tamamlandı*
* [ ] Faz 3: Güvenli Otonomi ve Model Tabanlı Zeka (Koopman/DMD & CBF)
* [ ] Faz 4: Görüş Ötesi Çatışma (Defansif BVR)
* [ ] Faz 5: Taktiksel İt Dalaşı ve Filo Koordinasyonu (MARL & Self-Play)

## 🛠️ Teknoloji Yığını ve Sistem Gereksinimleri

Bu projenin stabil çalışması için çok spesifik kütüphane versiyonları gereklidir (NumPy 2.0 ve Setuptools 70+ krizlerini önlemek adına). Proje, Blackwell mimarisi (Örn: RTX 5070 Ti) dahil olmak üzere yeni nesil donanımlarla tam uyumlu hale getirilmiştir.

* **İşletim Sistemi:** Windows (Kullanıcı Ortamı)
* **Fizik Motoru:** JSBSim
* **Yapay Zeka:** PyTorch (cu128 Nightly Build for `sm_120`), Stable-Baselines3
* **Görselleştirme:** TensorBoard, Tacview (ACMI Telemetry)

## Önerilen Okuma Listesi
* [Hierarchical Reinforcement Learning for Air Combat at DARPA's AlphaDogfight Trials](https://arxiv.org/pdf/2105.00990)

* [A Deep Reinforcement Learning Control Approach for High-Performance Aircraft](https://link.springer.com/article/10.1007/s11071-023-08725-y)


* [Autonomous Dogfight Decision-Making for Air Combat Based on Reinforcement Learning with Automatic Opponent Sampling](https://www.mdpi.com/2226-4310/12/3/265)

* [Dogfight Simulation of Autonomous Swarm UAVs Based on Multi-Agent Deep Reinforcement Learning](https://www.sciepublish.com/article/pii/955)


### Kurulum (Conda Environment)

Proje, kütüphane çakışmalarını (NumPy 2.0 vb.) önlemek adına sabitlenmiş versiyonlarla çalışmaktadır. Özel donanımlar (Örn: RTX 50 Serisi Blackwell) için PyTorch Nightly build önerilir.

```bash
# 1. Conda ortamını oluşturun ve aktif edin
conda create -n bvr_ai python=3.10 -y
conda activate bvr_ai

# 2. Donanımınıza uygun PyTorch sürümünü kurun (Örnek: CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)

# 3. Kalan tüm bağımlılıkları requirements.txt üzerinden tek seferde kurun
pip install -r requirements.txt
```

# [EN]
Autonomous BVR Fighter AI ✈️🧠
This project is an artificial intelligence endeavor aimed at training an autonomous fighter pilot using the Soft Actor-Critic (SAC) algorithm. The environment leverages the JSBSim open-source flight dynamics engine to accurately simulate the complex aerodynamics and flight mechanics of an F-16 fighter jet.

🚀 Project Road Map & Milestones
To guide the agent from zero knowledge to Beyond Visual Range (BVR) combat capability, the project is structured around a highly staged Curriculum Learning framework.

[X] Phase 1: Aerodynamic Survival & Trimming - Completed

[X] Phase 2: Navigation & Flight Discipline (Command Following) - Completed

[ ] Phase 3: Safe Autonomy & Model-Based Intelligence (Koopman/DMD & CBF)

[ ] Phase 4: Beyond Visual Range Combat (Defensive BVR)

[ ] Phase 5: Tactical Dogfight & Fleet Coordination (MARL & Self-Play)

🛠️ Technology Stack & System Requirements
To ensure stability and prevent known dependency crises (such as the NumPy 2.0 and Setuptools 70+ conflicts), this project relies on strictly pinned library versions. The environment has been optimized for full compatibility with next-generation hardware, including the Blackwell architecture (e.g., RTX 5070 Ti).

OS: Windows (User Environment)

Flight Dynamics Engine: JSBSim

AI Framework: PyTorch (cu128 Nightly Build for sm_120), Stable-Baselines3

Visualization & Telemetry: TensorBoard, Tacview (ACMI Telemetry format)

📚 Recommended Reading List
Hierarchical Reinforcement Learning for Air Combat at DARPA's AlphaDogfight Trials

A Deep Reinforcement Learning Control Approach for High-Performance Aircraft

Autonomous Dogfight Decision-Making for Air Combat Based on Reinforcement Learning with Automatic Opponent Sampling

Dogfight Simulation of Autonomous Swarm UAVs Based on Multi-Agent Deep Reinforcement Learning

Installation (Conda Environment)
To avoid library conflicts, it is highly recommended to use the provided dependencies. For specialized hardware (e.g., RTX 50 Series Blackwell), installing the PyTorch Nightly build is advised.

```bash
# 1. Create and activate the Conda environment
conda create -n bvr_ai python=3.10 -y
conda activate bvr_ai

# 2. Install the PyTorch version suitable for your hardware (Example: CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install all remaining dependencies via requirements.txt
pip install -r requirements.txt
```
