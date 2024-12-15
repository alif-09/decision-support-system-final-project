import spacy

# Muat model en-core-web-sm
nlp = spacy.load('en_core_web_sm')

# Tampilkan detail model, termasuk versi
print(nlp.meta)
