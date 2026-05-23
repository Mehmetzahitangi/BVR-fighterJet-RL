# Otonom Savaş Pilotu AI (Autonomous BVR Fighter AI) ✈️🧠

Bu proje, F-16 savaş uçaklarının uçuş dinamiklerini **JSBSim** açık kaynaklı uçuş fizik motoru kullanarak simüle eden ve **Soft Actor-Critic (SAC)** algoritması ile otonom bir savaş pilotu yetiştirmeyi amaçlayan bir yapay zeka oluşturma çalışmasıdır.

## 🚀 Proje Aşamaları

Proje, ajanın sıfır bilgiden Görüş Ötesi Çatışma (BVR) yeteneğine ulaşması için çok aşamalı bir "Müfredatlı Öğrenme" (Curriculum Learning) konseptiyle tasarlanmıştır.

* **[X] Faz 1: Aerodinamik Hayatta Kalma (Trim & Survival)** - *Tamamlandı*
* **[X] Faz 2: Seyrüsefer ve Komut Takibi (Navigation & Command Following)
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
