from models.fm import ImageFlowMatcher
from models.class_cond_fm import ImageFlowMatcherClassCond
import torch
import argparse
from torchvision.utils import save_image
import os

def get_args():
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Generation args
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Directory to save generated images')
    parser.add_argument('--num_steps', type=int, default=2,
                        help='Number of steps for generation')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of channels in generated images')
    parser.add_argument('--height', type=int, default=28,
                        help='Height of generated images')
    parser.add_argument('--width', type=int, default=28,
                        help='Width of generated images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--digits', type=int, nargs='+', default=None,
                        help='Digits to generate, e.g. --digits 0 1 2 (class-conditional model only)')

    return parser.parse_args()

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'num_classes' in ckpt.get('hyper_parameters', {}):
        return ImageFlowMatcherClassCond.load_from_checkpoint(checkpoint_path, map_location=device)
    return ImageFlowMatcher.load_from_checkpoint(checkpoint_path, map_location=device)

def main():
    args = get_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    model.to(device)
    model.eval()

    is_conditional = isinstance(model, ImageFlowMatcherClassCond)
    if args.digits is not None and not is_conditional:
        raise ValueError('--digits requires a class-conditional checkpoint')

    digits = args.digits  # None means all / unconditional

    sample_idx = 0
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - i)

        generate_kwargs = dict(
            batch_size=batch_size,
            sample_image_size=(args.channels, args.height, args.width),
            num_steps=args.num_steps,
        )

        if is_conditional and digits is not None:
            # cycle through the requested digits to fill the batch
            labels = torch.tensor([digits[j % len(digits)] for j in range(sample_idx, sample_idx + batch_size)], device=device)
            generate_kwargs['y'] = labels

        with torch.no_grad():
            samples = model.generate(**generate_kwargs)

        for j, sample in enumerate(samples):
            save_image(sample, os.path.join(args.output_dir, f'sample_{sample_idx + j}.png'))

        sample_idx += batch_size

    print(f"Generated {args.num_samples} images in {args.output_dir}")

if __name__ == '__main__':
    main()
