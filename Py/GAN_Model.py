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
