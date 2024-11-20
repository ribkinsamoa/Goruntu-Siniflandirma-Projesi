import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Veri seti dizinleri
train_dir = r'C:\Users\can.deniz\Desktop\Yüksek düzey prog ödev\plates\train'  # Train klasörünün yolu
test_dir = r'C:\Users\can.deniz\Desktop\Yüksek düzey prog ödev\plates\test'  # Test klasörünün yolu

# Veri ön işleme
image_size = (150, 150)  # Görüntü boyutu
batch_size = 16

# Eğitim verilerini yükleme
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# CNN Modeli Oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # İki sınıf için sigmoid
])

# Modeli Derleme
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Modeli Eğitme
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10  # İsteğe bağlı olarak artırabilirsiniz
)

# Sonuçları Görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Test verilerini yükleme
test_images = []
test_labels = []  # Test etiketleri

for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalizasyon
        test_images.append(img_array)

        # Etiketleri belirleme (örneğin, dosya adından)
        if 'cleaned' in filename:
            test_labels.append(0)  # 0: cleaned
        elif 'dirty' in filename:
            test_labels.append(1)  # 1: dirty

# Test görüntülerini numpy dizisine çevirme
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Modeli Test Etme
predictions = model.predict(test_images)
predicted_classes = (predictions > 0.5).astype("int32")  # 0.5 eşik değeri ile sınıflandırma

# Başarı oranını hesaplama
correct_predictions = np.sum(predicted_classes.flatten() == test_labels)
accuracy = correct_predictions / len(test_labels)

# Sonuçları yazdırma
for i, prediction in enumerate(predicted_classes):
    print(f'Image: {os.listdir(test_dir)[i]}, Prediction: {"Cleaned" if prediction[0] == 0 else "Dirty"}')

