import torch
from models.kd_vgg import StudentVGG
from models.kd_unet import KDUnet
from models.diffusion import IIDM_Diffusion

def perform_dry_run():
    print("Initiating Architecture Dry Run...")

    print("Loading KD-VGG (Student)...")
    vgg = StudentVGG()
    
    print("Loading KD-UNet and Diffusion Framework...")
    unet = KDUnet(in_channels=3, out_channels=1)
    diffusion = IIDM_Diffusion(denoise_model=unet, timesteps=1000)

    batch_size = 2
    print(f"\nGenerating Dummy Batch (Size: {batch_size})...")
    
    dummy_optical = torch.randn(batch_size, 6, 256, 256)
    dummy_dem = torch.randn(batch_size, 1, 256, 256)
    dummy_canopy = torch.randn(batch_size, 1, 256, 256)
    dummy_carbon_target = torch.randn(batch_size, 1, 256, 256)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,))

    try:
        print("\nExecuting Forward Pass Integration...")
        
        vgg_features = vgg(dummy_optical)
        cond_features = vgg_features[-1] 
        print(f"-> KD-VGG extracted features shape: {cond_features.shape}")

        dummy_structural_base = torch.cat([dummy_dem, dummy_canopy], dim=1)
        
        dummy_noisy_carbon = torch.randn(batch_size, 1, 256, 256)
        unet_input = torch.cat([dummy_noisy_carbon, dummy_structural_base], dim=1)
        
        print("-> Passing data through KD-UNet...")
        predicted_noise = unet(unet_input, dummy_timesteps, cond_features)
        
        print(f"-> KD-UNet output shape: {predicted_noise.shape}")
        
        print("-> Testing Diffusion Loss Calculation...")
        loss = diffusion.p_losses(dummy_carbon_target, dummy_timesteps, cond_features, dummy_structural_base)
        
        print("\n✅ DRY RUN SUCCESSFUL!")
        print(f"Final Simulated Loss: {loss.item():.4f}")
        print("All tensor dimensions align perfectly across the IIDM pipeline.")

    except Exception as e:
        print("\n❌ DRY RUN FAILED. Tensor Mismatch Error:")
        print(e)

if __name__ == "__main__":
    perform_dry_run()
