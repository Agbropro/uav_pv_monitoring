model.py       → neural network
dataset.py     → dataset
preprocessing.py → transforms
trainer.py     → training
evaluator.py   → evaluation
predictor.py   → inference

                    ┌──────────────────┐
                    │ preprocessing.py │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    dataset.py   │
                    └────────┬─────────┘
                             │
                             ▼
                         DataLoader
                             │
                             ▼
                    ┌──────────────────┐
                    │     model.py     │
                    │  PVFaultModel    │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             trainer.py         evaluator.py
                    │
                    ▼
                 best.pt
                    │
                    ▼
              predictor.py
                    │
                    ▼
          fault type + confidence


TRAINING

model
  ↓
forward
  ↓
computation graph
  ↓
loss
  ↓
backward()
  ↓
gradients
  ↓
optimizer.step()


VALIDATION

model
  ↓
forward
  ↓
loss / predictions
  ↓
metrics

(no gradients)
(no backward)
(no optimizer update)

TRAINING FLOW 
                    DATASET
                       ↓
                 dataset.py
                       ↓
                  DataLoader
                       ↓
               preprocessing.py
                       ↓
                  MobileNet
                       ↓
                  trainer.py
                       ↓
                    best.pt

config
   ↓
preprocessing
   ↓
dataset
   ↓
model
   ↓
optimizer
   ↓
criterion
   ↓
trainer
   ↓
best.pt