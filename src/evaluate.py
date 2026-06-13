import torch
import json
from src.utils.metrics import calculate_metrics
from src.utils.visualization import save_plots

def run_evaluation(model, dataloader, device):
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            pred = model.diffusion.sample(batch['optical'], batch['structural'])
            gt = batch['carbon'].to(device)
            
            metrics = calculate_metrics(pred, gt)
            all_metrics.append(metrics)
            save_plots(pred, gt, "results/figures")
    
    with open("results/metrics.json", "w") as f:
        json.dump(all_metrics, f)
    print("Evaluation Complete. Results saved in results/")

if __name__ == "__main__":
    
    print("Run evaluation script via training monitor")
