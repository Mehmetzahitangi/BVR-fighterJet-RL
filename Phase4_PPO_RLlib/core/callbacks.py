import os
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# ======= BÜTÜN SORUNUN KAYNAĞI BUYDU =======
# İşçiler veriyi gizli Temp klasörlerine yazıyor, Komutan ise sizin klasörünüzde arıyordu.
# Şimdi dosyanın yolunu sizin bilgisayarınıza MUTLAK (Absolute) olarak ZİNCİRLİYORUZ.
# Herkes zorunlu olarak bu dosyaya yazıp, bu dosyadan okuyacak!
KASA_YOLU = "C:/Zahit/BVR_fighterJet_AI/Phase4_PPO_RLlib/results.txt"

class BVRPhaseCallback(DefaultCallbacks):
    
    def on_episode_end(self, *, episode, **kwargs):
        try:
            # 1. Puanları Al (İşçi Bilgisayarda Çalışır)
            if hasattr(episode, "get_return"): reward = episode.get_return()
            elif hasattr(episode, "total_reward"): reward = episode.total_reward
            else: reward = 0.0

            won = 1.0 if reward > 50.0 else 0.0
            
            try:
                ep_len = float(len(episode)) # Yeni Ray (v3) API'si uzunluğu böyle verir!
            except Exception:
                ep_len = float(getattr(episode, "length", getattr(episode, "t", 0.0)))

            print(f"[İSTİHBARAT] Görev Bitti! Toplam Puan: {reward:.1f} | Vuruş: {won}")

            # =================================================================
            # 2. TENSORBOARD'U GERİ GETİREN SİHİRLİ DOKUNUŞ (Eski Yöntem)
            # Ray bu değerleri otomatik toplayıp grafik pencerelerini açacak!
            # =================================================================
            if hasattr(episode, "custom_metrics"):
                episode.custom_metrics["1_IS_HIT_RATE"] = won
                episode.custom_metrics["2_MY_EPISODE_REWARD"] = float(reward)
                episode.custom_metrics["3_EPISODE_LEN_MEAN"] = ep_len
                episode.custom_metrics["curriculum_phase"] = getattr(self, "current_phase", 1.0)

            # =================================================================
            # 3. KIRMIZI TELEFON HATTI (Sadece Seviye Atlatmak İçin Kasaya Yaz)
            # =================================================================
            with open(KASA_YOLU, "a", encoding="utf-8") as f:
                f.write(f"{reward},{won},{ep_len}\n")
                
        except Exception as e:
            pass


    def on_train_result(self, *, algorithm, result, **kwargs):
        reward_val, win_rate, ep_len_val = 0.0, 0.0, 0.0
        
        # ZİNCİRLENMİŞ KASADAN OKU! (Komutanda Çalışır)
        if os.path.exists(KASA_YOLU):
            try:
                with open(KASA_YOLU, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                if lines:
                    rewards, hits, lens = [], [], []
                    for line in lines:
                        line = line.strip()
                        if line: 
                            try: # YENİ: Sadece bu satırı korumaya al!
                                parts = line.split(",")
                                if len(parts) == 3: # Sadece 3 parçalı (sağlam) verileri kabul et
                                    r, w, l = map(float, parts)
                                    rewards.append(r); hits.append(w); lens.append(l)
                            except Exception:
                                pass # Bozuk satırı yoksay ve okumaya devam et!
                
                    if rewards: # Eğer listeye sağlam veri girdiyse hesapla
                        reward_val = sum(rewards) / len(rewards)
                        win_rate = sum(hits) / len(hits)
                        ep_len_val = sum(lens) / len(lens)
                        
            except Exception as e:
                pass
            finally:
                try:
                    open(KASA_YOLU, "w").close()
                except Exception:
                    pass


        print("\n" + "▼"*50)
        print(f"[RADAR KÖPEĞİ (MUTLAK HAT)] Vuruş Oranı: {win_rate:.2f} | Puan: {reward_val:.2f} | Hayatta Kalma: {ep_len_val:.2f}")
        

        # --- 2. MÜFREDAT ZIRHI (Tecrübe Kilitli) ---
        current_phase = getattr(self, "current_phase", 1) 
        
        # Ray'den mevcut eğitim turunu (Iteration) alıyoruz
        iteration = result.get("training_iteration", 0)

        # Sadece belirli bir tecrübeye (tur sayısına) ulaştıktan sonra terfi edebilir!
        if current_phase == 1:
            # 1. Seviyede en az 10 tur (iteration) geçirmiş ve %70 başarı sağlamışsa
            if win_rate > 0.70 and iteration >= 10:
                current_phase = 2
                print("\n[AKADEMİ] TEBRİKLER! Ajan Seviye 2'ye (Hareketli Hedef) Terfi Etti!\n")
                
        elif current_phase == 2:
            # 2. Seviyede en az 25 tur geçirmiş ve %85 başarı sağlamışsa
            if win_rate > 0.85 and iteration >= 25:
                current_phase = 3
                print("\n[AKADEMİ] TEBRİKLER! Ajan Seviye 3'e (Ölümcül İt Dalaşı) Terfi Etti!\n")

        # Not: RL eğitiminde "Rütbe Düşürmek" ajanların kafasını karıştırır (Unlearning). 
        # Bu yüzden ajan zorlukta bocalasa bile onu o zorlukta bırakmak en iyisidir.
        self.current_phase = current_phase

        # --- 3. MATRUŞKA ZIRHI (Kutu Açma ve Seviyeyi Uçağa İletme) ---
        def update_env_phase(runner):
            def safe_set_phase(env_obj):
                if hasattr(env_obj, "set_phase"): env_obj.set_phase(self.current_phase)
                elif hasattr(env_obj, "unwrapped") and hasattr(env_obj.unwrapped, "set_phase"): env_obj.unwrapped.set_phase(self.current_phase)
                elif hasattr(env_obj, "get_sub_environments"): 
                    for sub in env_obj.get_sub_environments(): safe_set_phase(sub)
                elif hasattr(env_obj, "envs"):
                    for sub in env_obj.envs: safe_set_phase(sub)
                elif hasattr(env_obj, "env"): safe_set_phase(env_obj.env)

            if hasattr(runner, "env") and runner.env is not None: safe_set_phase(runner.env)
            elif hasattr(runner, "foreach_env"): runner.foreach_env(lambda env: safe_set_phase(env))

        if hasattr(algorithm, "env_runner_group") and algorithm.env_runner_group is not None:
            algorithm.env_runner_group.foreach_env_runner(update_env_phase)
        else:
            worker_group = algorithm.workers() if callable(algorithm.workers) else algorithm.workers
            worker_group.foreach_worker(update_env_phase)
            
        # =====================================================================
        # --- 4. TENSORBOARD'A GRAFİKLERİ ZORLA ÇİZDİRME ---
        # =====================================================================
        if "custom_metrics" not in result:
            result["custom_metrics"] = {}
            
        result["custom_metrics"]["1_IS_HIT_RATE"] = float(win_rate)
        result["custom_metrics"]["2_MY_EPISODE_REWARD"] = float(reward_val)
        result["custom_metrics"]["3_EPISODE_LEN_MEAN"] = float(ep_len_val)
        result["custom_metrics"]["curriculum_phase"] = float(self.current_phase)

        # =====================================================================
        # --- 5. BEYİN METRİKLERİNİ (ENTROPI & VF_LOSS) ESNEK TARAMA (FUZZY SEARCH) ---
        # =====================================================================
        # Bu fonksiyon Ray'in tüm gizli klasörlerini tarayıp içinde belirlediğimiz 
        # harfler geçen (örn: "ent") tüm metrikleri bulur.
        def find_fuzzy_metrics(d, keyword, results_list):
            if not isinstance(d, dict):
                return
            for k, v in d.items():
                # Eğer değer bir sayıysa ve aradığımız kelime anahtarın (k) içinde geçiyorsa:
                if isinstance(v, (int, float)) and keyword in k.lower():
                    results_list.append((k, float(v)))
                elif isinstance(v, dict):
                    find_fuzzy_metrics(v, keyword, results_list)

        ent_list = []
        vf_list = []
        
        # Sadece tam kelimeyi değil, içinde "ent" ve "vf" (veya loss) geçen HER ŞEYİ arıyoruz!
        find_fuzzy_metrics(result, "ent", ent_list)
        find_fuzzy_metrics(result, "vf", vf_list)

        # Ray'in kapıya koyduğu sahte 0.0'ları çöpe atıp sadece değerleri alıyoruz
        valid_ent = [val for key, val in ent_list if val is not None and val != 0.0]
        valid_vf = [val for key, val in vf_list if val is not None and val != 0.0]

        # Gerçek bir değer bulabildiysek sonuncusunu alıyoruz
        ent_val = valid_ent[-1] if valid_ent else 0.0
        vf_loss_val = valid_vf[-1] if valid_vf else 0.0

        # Bizim custom_metrics panelimize ekle
        result["custom_metrics"]["4_ENTROPY"] = ent_val
        result["custom_metrics"]["5_VF_LOSS"] = vf_loss_val

        # --- RADAR ÇIKTILARI (GİZLİ İSİMLERİ İFŞA ET) ---
        print(f"[METRİK RADARI] Bulunan 'ent' Adayları: {ent_list}")
        print(f"[METRİK RADARI] Bulunan 'vf' Adayları: {vf_list}")
        print(f"[BEYİN TOMOGRAFİSİ] Entropi: {ent_val:.4f} | Değer Kaybı (VF_Loss): {vf_loss_val:.4f}")
        print("▲"*50 + "\n")