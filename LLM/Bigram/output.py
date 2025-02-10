import sys
import time

from train import *

# Initialize the model
model = bigram().to(device)

# Load the model's state dictionary
state = torch.load('D:\\GPT2\\bible_wigets_inter100000.pth', map_location=device)
model_state_dict = state['model_state_dict']
model.load_state_dict(model_state_dict)
loss_value = state['loss']  # Extract the loss value
epoch = state.get('epoch', 'N/A')
print(f"Last loss value from training: {loss_value:.4f}")
print(f"Model was saved after iteration: {epoch}")

# Set model to evaluation mode
model.eval()

# Generate text
for _ in range(3):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start token
    generated_indices = model.generate(context, max_new_tokens=3000)[0].tolist()
    words = decode(generated_indices)
    for char in words:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)
    print('\n' + '-' * 80 + '\n')


"""
10.79585 M parameters
step 0: train loss 4.4859, val loss 4.4883
step 300: train loss 2.1264, val loss 2.2203
step 600: train loss 1.6281, val loss 1.8009
step 900: train loss 1.4066, val loss 1.6016
step 1200: train loss 1.2758, val loss 1.4945
step 1500: train loss 1.1942, val loss 1.4255
step 1800: train loss 1.1302, val loss 1.3710
step 2100: train loss 1.0864, val loss 1.3372
step 2400: train loss 1.0505, val loss 1.3036
step 2700: train loss 1.0225, val loss 1.2838
step 3000: train loss 0.9976, val loss 1.2557
step 3300: train loss 0.9828, val loss 1.2481
step 3600: train loss 0.9663, val loss 1.2377
step 3900: train loss 0.9501, val loss 1.2321
step 4200: train loss 0.9378, val loss 1.2316
step 4500: train loss 0.9235, val loss 1.2188
step 4800: train loss 0.9099, val loss 1.1993
It took 1972788.0136966705 to train
"""
