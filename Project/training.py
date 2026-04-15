import torch
import time
import copy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Executes the mini-batch training loop, handles the CPU-to-GPU data transfer,
    and tracks loss and accuracy across both training and validation sets.
    """
    print(f"--> Pushing model to {device}...")
    model = model.to(device)

    #Keep track of the metrics for our Seaborn and Matplotlib charts later
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    #Save the best version of the model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            #Iterate over data in mini-batches
            for inputs, labels in dataloader:
                #This is the physical handoff to the gpu (or CPU fallback)
                inputs = inputs.to(device)
                labels = labels.to(device)

                #Zero the parameter gradients before each batch
                optimizer.zero_grad()

                #Forward pass
                #Only track history if we are in the 'train' phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # Get the index of the highest probability
                    loss = criterion(outputs, labels)

                    #Backward pass + Optimize (ONLY in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #Accumulate the statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #Calculate the final math for this epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            #Save the metrics to our history dictionary
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                #If this is the best validation accuracy we save the weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    #Load the best weights back into the model before returning it
    model.load_state_dict(best_model_wts)
    return model, history