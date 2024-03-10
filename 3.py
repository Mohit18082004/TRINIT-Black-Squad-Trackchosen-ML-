import tree
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Specify the directory paths to your training and testing datasets
train_data_dir = 'C:/Users/pranav/AppData/Local/Temp/Rar$DRa5668.9783/India/train'
test_data_dir = 'C:/Users/pranav/AppData/Local/Temp/Rar$DRa5668.14951/India/test'

# Load and preprocess the dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224))

# Define the model architecture
input_shape = (224, 224,3)
base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(4, activation='softmax')(x)  # Assuming 4 classes for road damages
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
