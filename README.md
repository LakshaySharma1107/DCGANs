# DCGAN: Deep Convolutional Generative Adversarial Network

## Dataset Preprocessing

1. **Download the Dataset:**

   - This project uses the **CelebA** dataset. You can download it from: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

2. **Set the Dataset Path:**

   - Extract the dataset and set the `dataroot` variable in the code to point to the dataset folder:
     ```python
     dataroot = "path/to/img_align_celeba"
     ```

3. **Transformations Applied:**

   - Resize images to **64x64**.
   - Convert to **PyTorch tensors**.
   - Normalize pixel values to **[-1, 1]** for stable GAN training.

---

## Training the Model

1. **Install Dependencies:**

   ```bash
   pip install torch torchvision matplotlib numpy
   ```

2. **Run the Training Script:**

   ```bash
   python train_dcgan.py
   ```

   (Replace `train_dcgan.py` with the actual notebook/script file.)

3. **Training Details:**

   - Uses **Adam optimizer**.
   - **Batch size:** 128
   - **Latent vector size:** 100
   - **Epochs:** 10
   - Generates images at different epochs for visualization.

---

## Testing the Model

1. **Generate New Images:**

   - After training, use the generator to create new images:

   ```python
   noise = torch.randn(64, 100, 1, 1, device=device)
   fake_images = generator(noise)
   ```

2. **Save Generated Images:**

   ```python
   torchvision.utils.save_image(fake_images, 'generated_samples.png', normalize=True)
   ```

---

## Expected Outputs

- The model should generate **realistic-looking faces** after training.
- The generated images will be saved periodically during training.
- Sample output images will be available in `generated_samples.png`.


## References



&#x20;   DCGAN Paper: https\://arxiv.org/abs/1511.06434

&#x20;   PyTorch DCGAN Tutorial: https\://pytorch.org/tutorials/beginner/dcgan\_faces\_tutorial.html 

