import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

loaded_model = tf.keras.models.load_model("best_model.h5")

image = Image.open("data/Trouser-PNG-Clipart.png").resize((28, 28)).convert("L")

# Normalized image, converted to numpy array
image_normalized = np.array(image) / 255.0
image_resized = image_normalized.reshape((28, 28))

print("Resized image shape:", image_resized.shape)

predictions = loaded_model.predict(np.array([image_resized]))
# Interpret the predictions
predicted_class = np.argmax(predictions)

# Print the predicted class
print(f"Predicted Class: {predicted_class}")

plt.imshow(image, cmap='gray')
plt.title(f"Predicted Class: {predicted_class}")
plt.show()
