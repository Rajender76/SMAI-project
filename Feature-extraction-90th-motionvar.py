#!/usr/bin/env python
# coding: utf-8

# ### ANOTHER FEATURE EXTRACTION USING MOTION CATPURE
# 

# ### FEATURE EXTRACTION USING 90th PERCENTILE

# In[ ]:


import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm  # For progress tracking

# Mount Google Drive if using Colab
# from google.colab import drive
# drive.mount('/content/drive')

# Global constants
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
BATCH_SIZE = 16
DROPOUT_RATE = 0.5  # Increased dropout for regularization

# Enhanced transformation with stronger normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)  # Simple augmentation during feature extraction
])

def calculate_max_frames(video_paths, percentile=90):
    """Dynamically calculate max frames based on dataset"""
    lengths = []
    for path in tqdm(video_paths, desc="Calculating video lengths"):
        cap = cv2.VideoCapture(path)
        lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    return int(np.percentile(lengths, percentile))

def extract_frames(video_path, num_frames):
    """Improved frame extraction with time-based sampling and zero-padding"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Time-based sampling
    if frame_count <= num_frames:
        indices = list(range(frame_count))
    else:
        duration = frame_count / fps
        timestamps = np.linspace(0, duration, num_frames)
        indices = (timestamps * fps).astype(int)
        indices = np.clip(indices, 0, frame_count-1)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))

    cap.release()
    seq_len = len(frames)

    # Zero-padding instead of repeating frames
    if len(frames) < num_frames:
        padding = [torch.zeros(3, FRAME_HEIGHT, FRAME_WIDTH)] * (num_frames - len(frames))
        frames.extend(padding)

    return torch.stack(frames[:num_frames]), seq_len

class FeatureExtractor(nn.Module):
    """Regularized feature extractor with dropout"""
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            *list(base_model.children())[:-1],
            nn.Dropout(DROPOUT_RATE)
        )

    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)

def extract_and_save_features(video_path, output_dir, num_frames):
    """With feature existence check and error handling"""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, f"{video_name}_features.pt")

    if os.path.exists(save_path):
        return  # Skip already processed files

    try:
        frames, seq_len = extract_frames(video_path, num_frames)
        feature_extractor = FeatureExtractor().eval().to(device)
        with torch.no_grad():
            features = feature_extractor(frames.to(device))

        torch.save({
            'features': features.cpu(),
            'seq_len': seq_len
        }, save_path)

    except Exception as e:
        print(f"Failed to process {video_path}: {str(e)}")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize dataset and calculate dynamic frame count
train_video_dir = "/content/drive/MyDrive/CricShot10 dataset"
features_train_dir = "/content/drive/MyDrive/cricket_features/train-90"
CLASS_LABELS = ['cover', 'defense', 'flick', 'hook', 'late_cut',
                'lofted', 'pull', 'square_cut', 'straight', 'sweep']


def load_dataset(video_dir):
    """
    Loads video paths and labels based on folder structure.
    Returns:
      video_paths: list of video file paths
      labels: corresponding list of label indices
    """
    video_paths = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_LABELS):
        class_folder = os.path.join(video_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: folder {class_folder} does not exist.")
            continue
        for file_name in os.listdir(class_folder):
            if file_name.endswith(('.avi', '.mp4')):
                full_path = os.path.join(class_folder, file_name)
                video_paths.append(full_path)
                labels.append(class_idx)
    return video_paths, labels

# Load dataset and calculate dynamic frame count
video_paths, labels = load_dataset(train_video_dir)
max_frames = calculate_max_frames(video_paths, percentile=90)
print(f"Using dynamic frame count: {max_frames}")

# Process videos with progress tracking
for path in tqdm(video_paths, desc="Extracting features"):
    extract_and_save_features(path, features_train_dir, max_frames)

print("Feature extraction complete!")

# In[ ]:


import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import joblib
import torchvision

# In[ ]:


import os
import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_DIR = "/content/drive/MyDrive/cricket_features/train-90"

CLASSES_LIST = ["cover", "defense", "flick", "hook", "late_cut", "lofted", "pull", "square_cut", "straight", "sweep"]

all_files = []
all_labels = []

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES_LIST)

# Save it for future use
joblib.dump(label_encoder, "label_encoder.pkl")

for file_name in os.listdir(FEATURE_DIR):
    if file_name.endswith('.pt'):
        # Example: "square_cut_001_features.pt" → class name is "square_cut"
        base_name = file_name.split('_features.pt')[0]
        class_name = '_'.join(base_name.split('_')[:-1])
        if class_name in CLASSES_LIST:
            label = CLASSES_LIST.index(class_name)
            all_files.append(file_name)
            all_labels.append(label)
        else:
            print(f"Warning: {class_name} not found in class list. Skipping {file_name}.")

print("Total feature files found:", len(all_files))

# Split data into training, validation, and test sets (for example, 80/10/10)
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

print(f"Train files: {len(train_files)} | Val files: {len(val_files)} | Test files: {len(test_files)}")

# In[ ]:


class FeatureDataset(Dataset):
    def __init__(self, feature_dir, file_list, labels):
        """
        Args:
            feature_dir (str): Directory containing saved .pt feature files.
            file_list (list): List of feature filenames (with extension) found in the feature_dir.
            labels (list): Corresponding list of labels (as integers) for each file.
        """
        self.feature_dir = feature_dir
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.feature_dir, file_name)
        data = torch.load(file_path, weights_only=True)
        # Assume your .pt files are dictionaries with keys "features" and "seq_len"
        features = data['features']   # Tensor of shape (NUM_FRAMES, FEATURE_DIM)
        seq_len = data.get('seq_len', features.shape[0])  # Use recorded sequence length, if available
        label = self.labels[idx]
        # Optionally, you could also return seq_len if you plan to pack sequences in your LRCN.
        return features, seq_len, label


# In[ ]:


from torch.utils.data import DataLoader

BATCH_SIZE = 16

train_dataset = FeatureDataset(FEATURE_DIR, train_files, train_labels)
val_dataset   = FeatureDataset(FEATURE_DIR, val_files, val_labels)
test_dataset  = FeatureDataset(FEATURE_DIR, test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# In[1]:



import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class LRCN(nn.Module):
    def __init__(self, input_size=2048,
                 hidden_size=512,      # Reduced from 768 to 512
                 num_layers=1,         # Reduced from 2 to 1
                 num_classes=10,
                 dropout=0.7):         # Increased dropout
        super(LRCN, self).__init__()

        # Feature embedding with normalization
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,    # Now takes input from feature embedding
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),  # Layer normalization instead of batch norm
            nn.Dropout(dropout),        # Higher dropout
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        # L2 regularization will be applied through weight decay in optimizer

    def forward(self, x, lengths):
        # Apply feature embedding first
        x = self.feature_embedding(x)

        # Pack sequence0
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        attn_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)

        # Apply weighted sum
        weighted_output = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)

        # Apply classifier
        logits = self.classifier(weighted_output)
        return logits

# In[ ]:


# FEATURE_DIM=2048
# NUM_CLASSES=10
# NUM_FRAMES=40

FEATURE_DIM = 2048
NUM_CLASSES = 10
NUM_FRAMES = 95
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# In[ ]:


import torch.optim as optim


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Instantiate the model, loss function, and optimizer
model = LRCN(
    input_size=FEATURE_DIM,
    hidden_size=128,  # Reduced from 1024
    num_layers=2,     # Reduced from 3
    num_classes=NUM_CLASSES,
    dropout=0.3      # Increased from 0.5
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY  # Added weight decay
)

# Learning rate scheduler - reduces learning rate when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # Monitor validation loss for improvement
    factor=0.5,       # Multiply learning rate by this factor when reducing
    patience=5,       # Wait for 5 epochs with no improvement before reducing LR
    verbose=True      # Print message when LR is reduced
)

# Early stopping parameters
best_val_loss = float('inf')
early_stopping_counter = 0
best_model_path = 'best_lrcn_model.pth'

# Training and validation loop
for epoch in range(EPOCHS):
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Training loop
    for features, seq_len, labels in train_loader:
        # features shape: (batch, NUM_FRAMES, FEATURE_DIM)
        features = features.to(device)
        labels = labels.to(device)

        # Convert sequence lengths tensor
        lengths = torch.tensor([min(seq_len_i, NUM_FRAMES) for seq_len_i in seq_len], dtype=torch.int64)
        lengths = lengths.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, lengths)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        train_loss += loss.item() * features.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data).item()
        train_total += features.size(0)

    # Calculate epoch metrics
    train_loss_epoch = train_loss / train_total
    train_acc = train_correct / train_total

    # VALIDATION PHASE
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for features, seq_len, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            lengths = torch.tensor([min(seq_len_i, NUM_FRAMES) for seq_len_i in seq_len], dtype=torch.int64)
            lengths = lengths.to(device)

            outputs = model(features, lengths)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_total += features.size(0)

    val_loss_epoch = val_loss / val_total
    val_acc = val_correct / val_total

    # Print metrics
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss {train_loss_epoch:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss_epoch:.4f}, Val Acc {val_acc:.4f}")

    # Learning rate scheduling
    scheduler.step(val_loss_epoch)

    # Early stopping check
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        early_stopping_counter = 0
        # Save best model
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print("Training completed!")

# ### ANOTHER FEATURE EXTRACTION USING MOTION CATPURE
# 

# 

# In[ ]:




# ### MOTION BASED FEATURE EXTRACTION

# In[ ]:


import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# In[ ]:


import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# Global constants
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_FRAMES = 40
BATCH_SIZE = 16
MOTION_THRESHOLD = 30  # threshold for frame difference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-load the feature extractor
feature_extractor = models.resnet50(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval().to(DEVICE)

# Define image transforms
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

transform = get_transform()

def sample_key_frames(video_path, num_frames=NUM_FRAMES):
    """
    Sample frames based on motion: picks frames with significant difference,
    then uniform-samples to fixed length.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    diffs = []
    prev_gray = None

    # Read all frames first (or up to a cap)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(np.mean(diff))
            frames.append(rgb)
        else:
            # first frame, always include
            diffs.append(0)
            frames.append(rgb)
        prev_gray = gray
    cap.release()

    if len(frames) == 0:
        # blank placeholder
        return [np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)] * num_frames

    # Compute top-k motion frames
    motion_indices = np.argsort(diffs)[-num_frames:]
    # If fewer, pad by repeating last
    if len(motion_indices) < num_frames:
        motion_indices = list(motion_indices) + [motion_indices[-1]] * (num_frames - len(motion_indices))

    # Uniformly select among these key frames if too many
    motion_indices = sorted(motion_indices)
    if len(motion_indices) > num_frames:
        motion_indices = np.linspace(0, len(motion_indices)-1, num_frames, dtype=int)
        motion_indices = [motion_indices[i] for i in range(num_frames)]

    sampled = [frames[i] for i in motion_indices]
    return sampled


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """
    Extract and transform frames using motion-aware sampling.
    Returns torch.Tensor (num_frames, 3, H, W) and seq_len.
    """
    raw_frames = sample_key_frames(video_path, num_frames)
    transformed = []
    for frame in raw_frames:
        try:
            t = transform(frame)
        except Exception:
            # fallback blank
            t = torch.zeros(3, FRAME_HEIGHT, FRAME_WIDTH)
        transformed.append(t)
    seq_len = len(transformed)
    frames_tensor = torch.stack(transformed)
    return frames_tensor, seq_len


def extract_features(frames, batch_size=BATCH_SIZE):
    """
    Extract CNN features on device, chunked in batches.
    """
    features = []
    with torch.no_grad():
        for i in range(0, frames.size(0), batch_size):
            batch = frames[i:i+batch_size].to(DEVICE)
            out = feature_extractor(batch)
            out = out.view(out.size(0), -1)
            features.append(out.cpu())
    return torch.cat(features, dim=0)


def extract_and_save_features(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames, seq_len = extract_frames(video_path)
    feats = extract_features(frames)
    save_dict = {'features': feats, 'seq_len': seq_len}
    save_path = os.path.join(output_dir, f"{video_name}_features.pt")
    torch.save(save_dict, save_path)
    print(f"Saved: {video_name}, seq_len={seq_len}")

if __name__ == '__main__':
    train_video_dir = "/content/drive/MyDrive/CricShot10 dataset"
    features_train_dir = "/content/drive/MyDrive/cricket_features/train-new"

    from glob import glob
    train_paths = [p for p in glob(os.path.join(train_video_dir, '*', '*')) if p.endswith(('.avi', '.mp4'))]
    print(f"Found {len(train_paths)} videos. Using device: {DEVICE}")

    for vp in tqdm(train_paths, desc="Extracting features"):
        extract_and_save_features(vp, features_train_dir)
    print("Feature extraction complete.")

# In[ ]:


import os
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import joblib
import torchvision

FEATURE_DIR = "/content/drive/MyDrive/cricket_features/train-new"

CLASSES_LIST = ["cover", "defense", "flick", "hook", "late_cut", "lofted", "pull", "square_cut", "straight", "sweep"]

all_files = []
all_labels = []

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES_LIST)

# Save it for future use
joblib.dump(label_encoder, "label_encoder.pkl")

for file_name in os.listdir(FEATURE_DIR):
    if file_name.endswith('.pt'):
        # Example: "square_cut_001_features.pt" → class name is "square_cut"
        base_name = file_name.split('_features.pt')[0]
        class_name = '_'.join(base_name.split('_')[:-1])
        if class_name in CLASSES_LIST:
            label = CLASSES_LIST.index(class_name)
            all_files.append(file_name)
            all_labels.append(label)
        else:
            print(f"Warning: {class_name} not found in class list. Skipping {file_name}.")

print("Total feature files found:", len(all_files))

# Split data into training, validation, and test sets (for example, 80/10/10)
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

print(f"Train files: {len(train_files)} | Val files: {len(val_files)} | Test files: {len(test_files)}")

# In[ ]:


class FeatureDataset(Dataset):
    def __init__(self, feature_dir, file_list, labels):
        """
        Args:
            feature_dir (str): Directory containing saved .pt feature files.
            file_list (list): List of feature filenames (with extension) found in the feature_dir.
            labels (list): Corresponding list of labels (as integers) for each file.
        """
        self.feature_dir = feature_dir
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.feature_dir, file_name)
        data = torch.load(file_path, weights_only=True)
        # Assume your .pt files are dictionaries with keys "features" and "seq_len"
        features = data['features']   # Tensor of shape (NUM_FRAMES, FEATURE_DIM)
        seq_len = data.get('seq_len', features.shape[0])  # Use recorded sequence length, if available
        label = self.labels[idx]
        # Optionally, you could also return seq_len if you plan to pack sequences in your LRCN.
        return features, seq_len, label


# In[ ]:


from torch.utils.data import DataLoader

BATCH_SIZE = 16

train_dataset = FeatureDataset(FEATURE_DIR, train_files, train_labels)
val_dataset   = FeatureDataset(FEATURE_DIR, val_files, val_labels)
test_dataset  = FeatureDataset(FEATURE_DIR, test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# In[ ]:



import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class LRCN(nn.Module):
    def __init__(self, input_size=2048,
                 hidden_size=512,      # Reduced from 768 to 512
                 num_layers=2,         # Reduced from 2 to 1
                 num_classes=10,
                 dropout=0.5):         # Increased dropout
        super(LRCN, self).__init__()

        # Feature embedding with normalization
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.6),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.6),
            nn.ReLU()
        )


        self.lstm = nn.LSTM(
            input_size=hidden_size,    # Now takes input from feature embedding
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),  # Layer normalization instead of batch norm
            nn.Dropout(0.8),        # Higher dropout
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        # L2 regularization will be applied through weight decay in optimizer

    def forward(self, x, lengths):
        # Apply feature embedding first
        x = self.feature_embedding(x)

        # Pack sequence0
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        attn_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)

        # Apply weighted sum
        weighted_output = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)

        # Apply classifier
        logits = self.classifier(weighted_output)
        return logits

# In[ ]:


# FEATURE_DIM=2048
# NUM_CLASSES=10
# NUM_FRAMES=40

FEATURE_DIM = 2048
NUM_CLASSES = 10
NUM_FRAMES = 40
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# In[ ]:


import torch.optim as optim


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Instantiate the model, loss function, and optimizer
model = LRCN(
    input_size=FEATURE_DIM,
    hidden_size=128,  # Reduced from 1024
    num_layers=2,     # Reduced from 3
    num_classes=NUM_CLASSES,
    dropout=0.3      # Increased from 0.5
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY  # Added weight decay
)

# Learning rate scheduler - reduces learning rate when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # Monitor validation loss for improvement
    factor=0.5,       # Multiply learning rate by this factor when reducing
    patience=5,       # Wait for 5 epochs with no improvement before reducing LR
    verbose=True      # Print message when LR is reduced
)

# Early stopping parameters
best_val_loss = float('inf')
early_stopping_counter = 0
best_model_path = 'best_lrcn_model.pth'

# Training and validation loop
for epoch in range(EPOCHS):
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Training loop
    for features, seq_len, labels in train_loader:
        # features shape: (batch, NUM_FRAMES, FEATURE_DIM)
        features = features.to(device)
        labels = labels.to(device)

        # Convert sequence lengths tensor
        lengths = torch.tensor([min(seq_len_i, NUM_FRAMES) for seq_len_i in seq_len], dtype=torch.int64)
        lengths = lengths.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, lengths)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        train_loss += loss.item() * features.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data).item()
        train_total += features.size(0)

    # Calculate epoch metrics
    train_loss_epoch = train_loss / train_total
    train_acc = train_correct / train_total

    # VALIDATION PHASE
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for features, seq_len, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            lengths = torch.tensor([min(seq_len_i, NUM_FRAMES) for seq_len_i in seq_len], dtype=torch.int64)
            lengths = lengths.to(device)

            outputs = model(features, lengths)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_total += features.size(0)

    val_loss_epoch = val_loss / val_total
    val_acc = val_correct / val_total

    # Print metrics
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss {train_loss_epoch:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss_epoch:.4f}, Val Acc {val_acc:.4f}")

    # Learning rate scheduling
    scheduler.step(val_loss_epoch)

    # Early stopping check
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        early_stopping_counter = 0
        # Save best model
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print("Training completed!")

# In[ ]:



