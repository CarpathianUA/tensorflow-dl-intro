import platform
import pickle
import time
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
# use legacy optimizer since a main one works slowly on Apple M1/M2
from tensorflow.keras.optimizers.legacy import Adam

print(f"Python Platform: {platform.platform()}")
print("TensorFlow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0


# Define a CNN model
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(10)(x)

# Create the CNN model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Define a ModelCheckpoint callback to save the best model during training
checkpoint_callback = ModelCheckpoint(
    "best_model.h5",  # File to save the best model
    save_best_only=True,  # Save only the best model
    monitor='val_loss',  # Monitor validation loss for saving
    mode='min',  # Minimize the monitored quantity (val_loss)
    verbose=1  # Display progress during training
)

# Train the model for a specified duration (e.g., 10 minutes)
# You can set the training time based on your requirements
max_training_time_minutes = 10
max_training_epochs = 20  # Adjust this as needed


# Callback to stop training after a certain amount of time
class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_minutes):
        self.max_minutes = max_minutes
        super(TimeoutCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (time.time() - self.model.start_time) / 60.0 >= self.max_minutes:
            self.model.stop_training = True


model.start_time = time.time()
timeout_callback = TimeoutCallback(max_minutes=max_training_time_minutes)

# Train the model using checkpoints and the timeout callback
print("Start time:",  datetime.utcfromtimestamp(model.start_time))
history = model.fit(
    X_train, y_train,
    epochs=max_training_epochs,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, timeout_callback]
)
print("End time:", datetime.now())
print("Training time:", datetime.now() - datetime.utcfromtimestamp(model.start_time))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

# Save training history to a file
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
