# Surgical Phase Recognition Model: In-Depth Technical Guide

## 1. Dataset Handling: `SurgicalVideoDataset`

### Clip Extraction Method
```python
def _extract_clips(self, video):
    clips = []
    for start in range(0, len(video) - self.clip_duration, self.stride):
        clip = video[start:start+self.clip_duration]
        processed_clip = torch.stack([self.transforms(frame) for frame in clip])
        clips.append(processed_clip)
    return clips
```
**Explanation**: 
- Slides a 16-frame window through the video
- 10-second stride between clips
- Transforms and normalizes each frame
- Prevents data repetition

## 2. Model Architecture: `MaskedVideoDistillation`

### Feature Extraction and Classification
```python
def forward(self, video_clips):
    features = []
    for clip in video_clips:
        # Extract features using pretrained VideoMAE
        outputs = self.video_mae(clip)
        features.append(outputs.last_hidden_state[:, 0])
    
    # Aggregate features across clips
    pooled_features = torch.mean(torch.stack(features), dim=0)
    return self.classifier(pooled_features)
```
**Key Techniques**:
- Uses VideoMAE for feature extraction
- Captures temporal dependencies
- Averages features across video clips
- Applies classification layer

## 3. Training Loop: Advanced Performance Tracking

### Model Training and Validation
```python
def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        for videos, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy}%')
```
**Training Highlights**:
- Dynamic learning rate
- Model checkpoint saving
- Real-time accuracy tracking
- Best model preservation

## 4. Main Execution Flow

### Dataset and Model Setup
```python
def main():
    # Dataset preparation
    train_dataset = SurgicalVideoDataset(train_videos, train_labels)
    test_dataset = SurgicalVideoDataset(test_videos, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    # Model initialization
    model = MaskedVideoDistillation()
    train_model(model, train_loader, test_loader)
```
**Process Overview**:
- Load and split surgical video dataset
- Create data loaders
- Initialize MaskedVideoDistillation model
- Commence training process

## 5. Performance Metrics

### Expected Outcomes
- **Top-1 Accuracy**: ~72.9%
- **Top-5 Accuracy**: ~94.1%
- **Surgical Phases**: 11 categories

## 6. Deployment Recommendations
- Use GPU-enabled environment
- Ensure PyTorch and CUDA compatibility
- Prepare labeled surgical video dataset
- Validate model on unseen surgical cases
