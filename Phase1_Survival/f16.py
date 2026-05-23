import jsbsim
# Simülatörü başlat ve F-16'yı yükle
fdm = jsbsim.FGFDMExec(None)
fdm.load_model('f16')


# F-16'nın okuyabileceğimiz ve müdahale edebileceğimiz tüm özelliklerini listeler
#print(fdm.get_property_catalog())

# İçinde "attitude" kelimesi geçen tüm değişkenleri listele
acilar = [prop for prop in fdm.get_property_catalog() if 'attitude' in prop]
for a in acilar:
    print(a)