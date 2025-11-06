# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load and Preprocess Dataset
data = pd.read_csv('NSL_KDD.csv')  # Replace with actual dataset path

# Drop irrelevant features
data.drop(['id', 'timestamp', 'src_ip', 'dst_ip'], axis=1, inplace=True)

# Encode categorical features
label_enc = LabelEncoder()
data['protocol'] = label_enc.fit_transform(data['protocol'])

# Normalize numerical data
scaler = MinMaxScaler()
features = data.drop('label', axis=1)
scaled_features = scaler.fit_transform(features)

# Binary classification: attack = 1, normal = 0
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
labels = data['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Step 3: Define Generator
def build_generator(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    return model

# Step 4: Define Discriminator
def build_discriminator(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Step 5: Train GAN
input_dim = X_train.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)

discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combined model
discriminator.trainable = False
gan_input = layers.Input(shape=(input_dim,))
generated_data = generator(gan_input)
validity = discriminator(generated_data)

gan = models.Model(gan_input, validity)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training loop
epochs = 1000
batch_size = 64

for epoch in range(epochs):
    # Generate fake data
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    fake_data = generator.predict(noise)

    # Select real data
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_data = X_train[idx]

    # Labels
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_data, valid)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake)

    # Train generator
    g_loss = gan.train_on_batch(noise, valid)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | D Loss: {np.mean([d_loss_real[0], d_loss_fake[0]]):.4f} | G Loss: {g_loss:.4f}")

# Step 6: Train Classifier on Combined Data
# Append synthetic data to training set
synthetic_data = generator.predict(np.random.normal(0, 1, (len(X_train)//2, input_dim)))
X_combined = np.vstack((X_train, synthetic_data))
y_combined = np.hstack((y_train, np.ones(len(synthetic_data))))  # Label fake data as attacks (1)

# Train simple classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_combined, y_combined)

# Step 7: Evaluate Classifier
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
