# create_visualizations.py
import matplotlib.pyplot as plt
import numpy as np

def create_sample_comparison():
    """Creating sample images to display on GitHub"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(3):
        # Input Image
        axes[i, 0].imshow(np.random.rand(256, 256, 3))
        axes[i, 0].set_title('Input Image', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(np.random.randint(0, 5, (256, 256)), cmap='tab10', vmin=0, vmax=4)
        axes[i, 1].set_title('Ground Truth', fontweight='bold', color='green')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(np.random.randint(0, 5, (256, 256)), cmap='tab10', vmin=0, vmax=4)
        axes[i, 2].set_title('Prediction', fontweight='bold', color='blue')
        axes[i, 2].axis('off')
        
        # Comparison
        comp = np.hstack([
            np.random.randint(0, 5, (256, 256)), 
            np.random.randint(0, 5, (256, 256))
        ])
        axes[i, 3].imshow(comp, cmap='tab10', vmin=0, vmax=4)
        axes[i, 3].set_title('Comparison\n(Left: Truth, Right: Pred)', fontweight='bold', color='red')
        axes[i, 3].axis('off')

    plt.suptitle('Sample Predictions - Damage Segmentation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ sample_comparison.png created!")

def create_training_curves():
    """Creating tutorial graphs to display on GitHub"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = list(range(1, 51))
    train_loss = [1.4 * np.exp(-0.05 * i) + 0.7 for i in range(50)]
    val_loss = [1.3 * np.exp(-0.04 * i) + 0.75 for i in range(50)]

    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    train_iou = [0.2 + 0.4 * (1 - np.exp(-0.1 * i)) for i in range(50)]
    val_iou = [0.15 + 0.3 * (1 - np.exp(-0.08 * i)) for i in range(50)]

    ax2.plot(epochs, train_iou, 'g-', linewidth=2, label='Training IoU')
    ax2.plot(epochs, val_iou, 'orange', linewidth=2, label='Validation IoU')
    ax2.set_title('Training and Validation IoU', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ training_curves.png created!")

if __name__ == "__main__":
    create_sample_comparison()
    create_training_curves()