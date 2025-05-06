#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam



# In[2]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# In[4]:


# Set random seeds for reproducibility

# In[11]:


NUM_FRAMES = 30      # Temporal dimension
FRAME_SIZE = 112         # Spatial dimension (reduced for memory)
BATCH_SIZE = 8            # Balance between memory and batch normalization
EPOCHS = 40               # Allow for potential longer training with callbacks
LEARNING_RATE = 1e-4      # Initial learning rate
CLASSES = ['cover', 'defense', 'flick', 'hook', 'late_cut', 'lofted', 'pull', 'square_cut', 'straight', 'sweep']
NUM_CLASSES = len(CLASSES)

# In[38]:


tf.keras.mixed_precision.set_global_policy('mixed_float16')

# In[39]:


import tensorflow as tf

def pad_last_batch(frames, target_size=30):
    """Pad the last batch to the required target size"""
    current_size = tf.shape(frames)[0]  # Get current number of frames
    if current_size < target_size:
        # Calculate how many frames we need to add
        padding_size = target_size - current_size
        # Pad the frames
        frames = tf.concat([frames, tf.zeros([padding_size, *frames.shape[1:]], dtype=frames.dtype)], axis=0)
    return frames

# In[40]:


def frame_augmentation(frame):
    """Spatial data augmentation for individual frames"""
    frame = tf.image.random_brightness(frame, 0.2)
    frame = tf.image.random_contrast(frame, 0.8, 1.2)
    frame = tf.image.random_flip_left_right(frame)
    return frame

# In[41]:


def temporal_augmentation(frames):
    """Temporal data augmentation for video sequences"""
    # Apply augmentation steps like cropping (optional)
    if tf.random.uniform(()) > 0.5:
        start = tf.random.uniform((), 0, NUM_FRAMES // 4, dtype=tf.int32)
        frames = frames[start:start + NUM_FRAMES]

    # Ensure frames are the correct size
    if len(frames) < NUM_FRAMES:
        frames = tf.image.resize(frames, (FRAME_SIZE, FRAME_SIZE))  # Resize frames spatially

    # Pad the batch to the target size if necessary
    frames = pad_last_batch(frames, target_size=30)

    return frames


# In[42]:


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Optimized frame extraction with memory management"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // num_frames  # Frame sampling interval

    # Sample frames efficiently based on frame_interval
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)
        else:
            frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3)))  # Padding if needed

    cap.release()
    return np.array(frames, dtype=np.float32)


def create_dataset(video_paths, labels, is_training=False):
    """Optimized tf.data pipeline with augmentation for 3D CNNs"""
    def process_path(path, label):
        # Load and preprocess video
        frames = tf.numpy_function(
            lambda x: extract_frames(x.decode('utf-8')),
            [path],
            Tout=tf.float32
        )
        frames.set_shape((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3))
        
        # MobileNetV2 preprocessing (can also be replaced with preprocessing for 3D CNNs if available)
        frames = tf.keras.applications.mobilenet_v2.preprocess_input(frames)
        
        # Augmentations for 3D CNN
        if is_training:
            frames = temporal_augmentation(frames)  # Temporal augmentations (random time shifts, etc.)
            frames = tf.map_fn(frame_augmentation, frames)  # Spatial augmentations (random flips, etc.)
        
        return frames, tf.one_hot(label, NUM_CLASSES)
    
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    if is_training:
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# In[ ]:



def improved_extract_frames(video_path):
    """More robust frame extraction with motion-based keyframe selection"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    
    while len(frames) < NUM_FRAMES and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        
        # Simple motion detection - only keep frames with significant changes
        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            if np.mean(diff) > 15:  # Motion threshold
                frames.append(frame)
        else:
            frames.append(frame)
            
        prev_frame = frame
    
    cap.release()
    
    # Smart padding with last frame repetition
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1] if len(frames) > 0 else np.zeros((FRAME_SIZE, FRAME_SIZE, 3)))
    
    # Normalization with EfficientNet preprocessing
    frames = tf.keras.applications.efficientnet.preprocess_input(np.array(frames))
    return frames

# In[ ]:



def create_optimized_dataset(video_paths, labels, is_training=False):
    """Enhanced data pipeline with better augmentation"""
    def process_path(path, label):
        frames = tf.numpy_function(
            lambda x: improved_extract_frames(x.decode('utf-8')),
            [path],
            Tout=tf.float32
        )
        frames.set_shape((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3))
        
        # Stronger augmentation for training
        if is_training:
            frames = tf.image.random_brightness(frames, 0.2)
            frames = tf.image.random_contrast(frames, 0.8, 1.2)
            if tf.random.uniform(()) > 0.5:
                frames = tf.image.flip_left_right(frames)
        
        return frames, tf.one_hot(label, NUM_CLASSES)
    
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    if is_training:
        dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)
    
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Training setup
model = build_enhanced_model()
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Create datasets
train_ds = create_optimized_dataset(train_paths, train_labels, is_training=True)
val_ds = create_optimized_dataset(val_paths, val_labels)

# Train with class weights
class_counts = np.bincount(train_labels)
total_samples = sum(class_counts)
class_weights = {i: total_samples/(len(class_counts)*count) for i, count in enumerate(class_counts)}

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# In[43]:


def build_enhanced_model():
    """Optimized architecture with better feature extraction"""
    # Base CNN Model with proper initialization
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
        pooling='avg'  # Add global pooling directly
    )
    
    # Freezing strategy
    for layer in base_model.layers[:150]:
        layer.trainable = False
    for layer in base_model.layers[150:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    
    # Input layer
    inputs = layers.Input(shape=(NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3))
    
    # Frame processing
    x = layers.TimeDistributed(base_model)(inputs)
    
    # Enhanced temporal processing
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.4)(x)
    
    # Classification head with proper initialization
    x = layers.Dense(256, activation='relu', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Custom learning rate with warmup
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-5,  # Lower initial LR
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9
    )
    
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# In[ ]:


# Training setup
model = build_enhanced_model()
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Create datasets
train_ds = create_optimized_dataset(train_paths, train_labels, is_training=True)
val_ds = create_optimized_dataset(val_paths, val_labels)

# Train with class weights
class_counts = np.bincount(train_labels)
total_samples = sum(class_counts)
class_weights = {i: total_samples/(len(class_counts)*count) for i, count in enumerate(class_counts)}

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
        class_weight=class_weights
)

# In[44]:


DATA_DIR = '/kaggle/input/cricshot-dataset/CricShot10 dataset'

# Load dataset paths
video_paths, labels = [], []
for class_idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for video_file in os.listdir(class_dir):
        if video_file.endswith('.avi'):
            video_paths.append(os.path.join(class_dir, video_file))
            labels.append(class_idx)

# Split dataset
train_paths, test_paths, train_labels, test_labels = train_test_split(
    video_paths, labels, test_size=0.2, stratify=labels, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, stratify=train_labels, random_state=42)

# Create datasets
train_ds = create_dataset(train_paths, train_labels, is_training=True)
val_ds = create_dataset(val_paths, val_labels)
test_ds = create_dataset(test_paths, test_labels)

# In[45]:


model = build_optimized_model()
model.summary()

# Enhanced callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# Class weighting for imbalanced data
class_counts = np.bincount(train_labels)
class_weights = {i: sum(class_counts)/count for i, count in enumerate(class_counts)}

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluation
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Generate classification report
y_pred = model.predict(test_ds)
y_true = np.concatenate([y for x, y in test_ds], axis=0)
print("\nClassification Report:")
print(classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), target_names=CLASSES))

# In[ ]:




# In[ ]:




# In[ ]:




# In[7]:


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Optimized frame extraction with circular buffer"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Circular buffer for efficient frame storage
    buffer = []
    for _ in range(min(frame_count, num_frames)):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            buffer.append(frame)
            if len(buffer) > num_frames:
                buffer.pop(0)
    
    # Pad if necessary
    while len(buffer) < num_frames:
        buffer.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3)))
    
    cap.release()
    return np.array(buffer, dtype=np.float32) / 255.0

# In[8]:


def create_dataset(video_paths, labels):
    """Create optimized tf.data Dataset pipeline"""
    def load_video(path, label):
        frames = tf.numpy_function(
            lambda x: extract_frames(x.decode('utf-8')),
            [path],
            Tout=tf.float32
        )
        frames.set_shape((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3))
        return frames, tf.one_hot(label, NUM_CLASSES)
    
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.map(load_video, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# In[9]:


def build_efficient_model():
    """Memory-optimized model architecture"""
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(FRAME_SIZE, FRAME_SIZE, 3)
    )
    base_model.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=(NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        Bidirectional(LSTM(128, return_sequences=False)),  # Single LSTM layer
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax', dtype=tf.float32)
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# In[10]:


def load_dataset_paths(data_dir):
    video_paths = []
    labels = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        for video_file in os.listdir(class_dir):
            if video_file.endswith('.avi'):
                video_paths.append(os.path.join(class_dir, video_file))
                labels.append(class_idx)
    return video_paths, labels

# In[11]:


DATA_DIR = '/kaggle/input/cricshot-dataset/CricShot10 dataset'
video_paths, labels = load_dataset_paths(DATA_DIR)

# In[12]:


train_paths, test_paths, train_labels, test_labels = train_test_split(
    video_paths, labels, test_size=0.2, stratify=labels, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, stratify=train_labels, random_state=42)

# Create datasets
train_ds = create_dataset(train_paths, train_labels)
val_ds = create_dataset(val_paths, val_labels)
test_ds = create_dataset(test_paths, test_labels)

# In[13]:


model = build_efficient_model()
model.summary()

# In[16]:


callbacks = [
    
    ModelCheckpoint('best_model.keras',save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluation
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# In[4]:


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (uniformly distributed)
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Normalize pixel values
        frame = frame / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    # If we couldn't extract enough frames, pad with zeros
    while len(frames) < num_frames:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))
    
    return np.array(frames)

# Function to load dataset
def load_dataset(data_dir):
    """Load video data and labels from directory."""
    X = []  # Will contain video frames
    y = []  # Will contain labels
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        print(f"Loading class {class_name} from {class_dir}")
        
        for video_file in os.listdir(class_dir):
            if video_file.endswith('.avi'):
                video_path = os.path.join(class_dir, video_file)
                frames = extract_frames(video_path)
                X.append(frames)
                y.append(class_idx)
    
    return np.array(X), np.array(y)

# Function to create a data generator for memory efficiency
def data_generator(X, y, batch_size=BATCH_SIZE):
    """Generator to yield batches of data."""
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    while True:
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X = np.array([X[i] for i in batch_indices])
            batch_y = np.array([y[i] for i in batch_indices])
            
            # One-hot encode the labels
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=NUM_CLASSES)
            
            yield batch_X, batch_y

# In[5]:


def build_lrcn_model(pretrained_model='mobilenetv2'):
    """Build an LRCN model using a pre-trained CNN."""
    # Define the input shape
    input_shape = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)
    
    # Create the base CNN model
    if pretrained_model.lower() == 'mobilenetv2':
        base_model = MobileNetV2(
            include_top=False, 
            weights='imagenet',
            input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3)
        )
        feature_size = 1280
    elif pretrained_model.lower() == 'resnet50':
        base_model = ResNet50(
            include_top=False, 
            weights='imagenet',
            input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3)
        )
        feature_size = 2048
    else:
        raise ValueError("Supported pretrained models are 'mobilenetv2' and 'resnet50'")
    
    # Freeze the weights of the base model
    base_model.trainable = False
    
    # Build the LRCN model
    model = Sequential()
    
    # TimeDistributed wrapper applies the CNN to each frame independently
    model.add(TimeDistributed(base_model, input_shape=input_shape))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    
    # LSTM layers for temporal sequence modeling
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(256, return_sequences=False)))
    model.add(Dropout(0.3))
    
    # Dense layers for classification
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# In[6]:


def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# In[7]:


DATA_DIR = '/kaggle/input/cricshot-dataset/CricShot10 dataset'  # Update this path to your dataset location
    
    # Load the dataset
print("Loading dataset...")
X, y = load_dataset(DATA_DIR)
print(f"Dataset loaded: {X.shape[0]} videos, each with {X.shape[1]} frames of shape {X.shape[2]}x{X.shape[3]}")
    
    # Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    
print(f"Training set: {X_train.shape[0]} videos")
print(f"Validation set: {X_val.shape[0]} videos")
print(f"Test set: {X_test.shape[0]} videos")
        
    # Create data generators
train_gen = data_generator(X_train, y_train, batch_size=BATCH_SIZE)
val_gen = data_generator(X_val, y_val, batch_size=BATCH_SIZE)
    
    # Calculate steps per epoch
steps_per_epoch = len(X_train) // BATCH_SIZE
validation_steps = len(X_val) // BATCH_SIZE
    
    # Build the model
model = build_lrcn_model(pretrained_model='resnet50')
model.summary()
    
    # Define callbacks
checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
print("Training the model...")
history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Plot training history
plot_training_history(history)
    
    # Evaluate on test set
print("Evaluating on test set...")
test_gen = data_generator(X_test, y_test, batch_size=BATCH_SIZE)
test_steps = len(X_test) // BATCH_SIZE
    
    # Evaluate the model
test_loss, test_acc = model.evaluate(test_gen, steps=test_steps)
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
    
    # Predict on test set
y_pred_probs = model.predict(np.array(X_test))
y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    # Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[4]:




# In[5]:




# In[6]:




# In[ ]:




# In[ ]:



