import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import os
import joblib # --- *** NEW: To save the scaler *** ---

# --- Constants ---
CSV_FILE = 'student_data.csv'
MODEL_FILE = 'attentiveness_model.keras'
SCALER_FILE = 'scaler.joblib' # --- *** NEW: Scaler file name *** ---
TIME_STEPS = 30  # How many frames to look back? (30 frames @ 30fps = 1 second)
N_FEATURES = 7   # yaw, pitch, roll, ear, mar, eb_ratio, gaze_ratio

# --- 1. Load Data ---
print(f"Loading data from {CSV_FILE}...")
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found. Please run data_collector.py to create it.")
    exit()

df = pd.read_csv(CSV_FILE)

# Handle potential NaN values (e.g., from failed pose detection)
df = df.fillna(0.0)

# Check if data is empty
if df.empty:
    print("Error: The data file is empty. Please collect some data first.")
    exit()

print(f"Loaded {len(df)} rows of data.")

# --- 2. Prepare Features (X) and Labels (y) ---
# Features are all columns except the last one ('label')
X = df.iloc[:, :N_FEATURES].values
# Label is the last column
y = df.iloc[:, -1].values

# --- 3. Scale Features ---
# Scaling is crucial for neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- *** NEW: Save the scaler *** ---
print(f"Saving the scaler to {SCALER_FILE}...")
joblib.dump(scaler, SCALER_FILE)
print("Scaler saved successfully.")
# --- *** END NEW *** ---

# --- 4. Create Temporal Sequences ---
# This is the most important step for an RNN/LSTM
# We convert flat data (rows) into sequences (rows, timesteps, features)
def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Get a "window" of features
        v = X[i:(i + time_steps)]
        # Get the label for that window (the label at the end of the window)
        l = y[i + time_steps]
        Xs.append(v)
        ys.append(l)
    return np.array(Xs), np.array(ys)

print(f"Creating sequences with {TIME_STEPS} time steps...")
X_seq, y_seq = create_sequences(X_scaled, y)

if X_seq.shape[0] == 0:
    print(f"Error: Not enough data to create sequences. Need at least {TIME_STEPS + 1} rows.")
    exit()

print(f"Original data shape: {X_scaled.shape}")
print(f"Sequential data shape: {X_seq.shape}") # (num_samples, TIME_STEPS, N_FEATURES)
print(f"Sequential labels shape: {y_seq.shape}") # (num_samples,)

# --- 5. Split Data ---
# Split the sequential data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# --- 6. Build the LSTM Model ---
print("Building the LSTM model...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, N_FEATURES)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification (0 or 1)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 7. Train the Model ---
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=20,          # How many times to see the data
    batch_size=32,      # How many samples to process at once
    validation_split=0.2, # Use part of training data for validation
    shuffle=True
)

print("Training complete.")

# --- 8. Evaluate the Model ---
print("Evaluating the model on the test set...")
# Predict probabilities (0.0 to 1.0)
y_pred_probs = model.predict(X_test)
# Convert probabilities to class labels (0 or 1)
y_pred = (y_pred_probs > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("------------------------\n")

# --- 9. Save the Model ---
print(f"Saving the trained model to {MODEL_FILE}...")
model.save(MODEL_FILE)
print("Model saved successfully.")
print("\n--- Next Step ---")
print("You can now run 'run_analyzer.py' to see the model in real-time!")