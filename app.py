import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from flask import Flask, request, render_template
from keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

# Set up paths
dataset_root = r'C:\Users\Dell\PycharmProjects\pythonProject6\cell_images'
parasitized_cell_path_prefix = os.path.join(dataset_root, 'Parasitized')
uninfected_cell_path_prefix = os.path.join(dataset_root, 'Uninfected')

# Get file paths
parasited_paths = [os.path.join(parasitized_cell_path_prefix, i) for i in os.listdir(parasitized_cell_path_prefix)]
uninfected_paths = [os.path.join(uninfected_cell_path_prefix, i) for i in os.listdir(uninfected_cell_path_prefix)]

# Create DataFrame
df_parasited = pd.DataFrame({'filename': parasited_paths, 'class': 'parasitized'})
df_uninfected = pd.DataFrame({'filename': uninfected_paths, 'class': 'uninfected'})
df = pd.concat([df_parasited, df_uninfected], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# ImageDataGenerator
batch_size = 8
target_size = (135, 135)
train_image_data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Flow from dataframe
train_generator = train_image_data_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    seed=42,
    subset='training'
)

validation_generator = train_image_data_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    seed=42,
    subset='validation'
)

# Model
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(135, 135, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plot_path = 'static/training_validation_accuracy.png'
plt.savefig(plot_path)

# Evaluate
test_images = test_df['filename']
test_labels = (test_df['class'] == 'uninfected').astype(int)
loaded_images = []
for path in test_images:
    img = cv2.imread(path)
    if img is None:
        print(f"Unable to read image at path: {path}")
    else:
        resized_img = cv2.resize(img, target_size)
        loaded_images.append(resized_img)
test_images_array = np.array(loaded_images)

# Check if the number of samples in input data and labels match
if len(test_images_array) != len(test_labels):
    print("Error: Number of samples in input data and labels don't match.")
else:
    # Evaluate the model
    accuracy = model.evaluate(test_images_array, test_labels)[1]
    print(f"Accuracy score is {accuracy}")

    # Serialize model architecture to JSON
    model_json = model.to_json()
    with open("models/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize model weights to HDF5
    model.save_weights(r"C:\Users\Dell\PycharmProjects\pythonProject6\models\model_weights.weights.h5")

app = Flask(__name__)

# Load the serialized model architecture
with open("models/model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the serialized model weights
loaded_model.load_weights(r"C:\Users\Dell\PycharmProjects\pythonProject6\models\model_weights.weights.h5")

def predict_class(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(135, 135))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict the class
    prediction = loaded_model.predict(img)
    if prediction[0] < 0.5:
        return 'Parasitized'
    else:
        return 'Uninfected'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the file
            file_path = 'static/uploads/' + file.filename
            file.save(file_path)

            # Predict the class
            predicted_class = predict_class(file_path)

            return render_template('result.html', file_path=file_path, predicted_class=predicted_class, accuracy=accuracy, plot_path=plot_path)
    return render_template('index.html')


@app.route('/graph')
def show_graph():
    return render_template('graph.html')

if __name__ == '__main__':
    app.run(debug=True)