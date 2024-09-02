import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Veri setini yükleme
data = pd.read_csv("veriseti.csv")

# Özellik çıkarma
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['category']

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-NN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)  # k değeri olarak 5 seçildi
knn.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = knn.predict(X_test)  # Test veri seti için tahminler
accuracy = accuracy_score(y_test, y_pred)  # Doğruluk oranı hesaplama
print("Accuracy:", accuracy)  # Doğruluk oranını yazdırma
print(classification_report(y_test, y_pred))  # Precision, Recall, F1-Skoru ve Destek raporunu yazdırma


# Modeli ve TF-IDF vektörizer'ı kaydetme
joblib.dump(knn, 'haber_kategori_knn_modeli.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("K-NN Model ve TF-IDF vektörizer başarıyla kaydedildi!")

# Eğitilmiş modeli ve TF-IDF vektörizer'ı yükleme
knn_model = joblib.load('haber_kategori_knn_modeli.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Yeni veri kümesini yükleme
new_data = pd.read_csv('haberler_icerik.csv', header=None, names=['title', 'content'])

# Metin verilerini TF-IDF vektörlerine dönüştürme
X_new = tfidf_vectorizer.transform(new_data['content'])

# Eğitilmiş modeli kullanarak tahminlerde bulunma
y_pred_new = knn_model.predict(X_new)

# Sonuçları kaydetme
new_data['predicted_category'] = y_pred_new

# Haber başlıkları ve içeriklerini de ekleyerek kaydetme
results = new_data[['content', 'predicted_category']]
results.to_csv('knn_kategori_tahmin_sonuclari.csv', index=False, encoding='utf-8')


print("Yeni veri kümesi başarıyla kategorilendirildi ve 'knn_kategori_tahmin_sonuclari.csv' dosyasına kaydedildi.")
