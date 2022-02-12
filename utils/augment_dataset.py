import argparse
from pathlib import Path
from typing import List, Tuple

import imgaug.augmenters as iaa
import numpy
from PIL import Image
from tqdm import tqdm


def augment_image(original_image: Image.Image, num_images: int) -> List[Image.Image]:
    assert original_image.mode == original_image.mode, "The given images have different color modes."
    # TODO: maybe have a look at this again
    geometric_aug = iaa.Sequential([
        iaa.SomeOf((1, 2), [
            iaa.ElasticTransformation(alpha=(5.0, 25.0), sigma=(5.0, 9.0)),
            iaa.ShearX((20, 20)),
            iaa.CropAndPad((-80, 80)),
            iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}),
        ]),
        iaa.Sometimes(0.66, iaa.OneOf([
            iaa.Rot90([1, 3]),
            iaa.Rotate((-40, 40))
        ])),
    ]).to_deterministic()

    color_aug = iaa.Sequential([
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.0, 1.0))),
        iaa.Sometimes(0.8, iaa.OneOf([
            iaa.GammaContrast((1.5, 2.5)),  # darker
            iaa.GammaContrast((0.1, 1.0)),  # lighter
        ])),
        iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)),
        iaa.Sometimes(0.10, iaa.Invert()),
    ]).to_deterministic()

    # Duplicates each image num_images times so that in the end each image can be augmented individually
    inflated_original_image = numpy.stack([numpy.array(original_image) for _ in range(num_images)])

    augmented_original_batch = geometric_aug(images=color_aug(images=inflated_original_image))
    augmented_original_images = [Image.fromarray(numpy.squeeze(arr))
                                 for arr in numpy.split(augmented_original_batch, num_images)]

    return list(augmented_original_images)


def save_image_batches(augmented_images: List[Tuple[Image.Image, Image.Image]], image_path: Path,
                       dataset_dir: Path, save_dir: Path):
    image_sub_dir = image_path.relative_to(dataset_dir).parent
    new_sub_dir = save_dir / image_sub_dir
    if not new_sub_dir.exists():
        new_sub_dir.mkdir(parents=True)
    for image_idx, orig_image in enumerate(augmented_images):
        output_image = Image.new('RGB', (orig_image.width, orig_image.height))
        output_image.paste(orig_image, (0, 0))
        output_image.save(new_sub_dir / f"{image_path.stem}_aug_{image_idx + 1}{image_path.suffix}")


def main(args: argparse.Namespace):
    image_paths = [f for f in args.dataset_dir.glob('**/*.jpg')]
    print(args.dataset_dir)
    for image_path in tqdm(image_paths[:10], desc="Processing image batches..."):
        image = Image.open(image_path)
        augmented_images = augment_image(image, num_images=args.num_augmented_images,
                                         show_images=args.show)
        if args.save_dir is not None:
            save_image_batches(augmented_images, image_path, args.dataset_dir, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script that produces an augmented dataset that can be used for "
                                                 "document segmentation. Augmentation operations may include "
                                                 "ElasticTransform, Shear, Crop, Shifting, Rotation, GammaContrast "
                                                 "and color inversion.")
    parser.add_argument("dataset_dir", type=Path,
                        help="Path of the top level directory of the dataset that should be augmented.")
    parser.add_argument("-n", "--num-augmented-images", type=int, default=3,
                        help="How many augmented images should be produced for every input image")
    parser.add_argument("--show", action="store_true", default=False, help="Shows generated images")
    parser.add_argument("--save-dir", type=Path, help="Saves generated images to the given directory.")
    main(parser.parse_args())
