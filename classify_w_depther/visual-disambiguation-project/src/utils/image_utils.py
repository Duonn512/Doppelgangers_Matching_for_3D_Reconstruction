def load_image(path):
    import cv2
    import numpy as np
    from PIL import Image

    cv_type = cv2.IMREAD_COLOR
    image = None
    if str(path).endswith('gif'):
        pil_image = Image.open(str(path))
        if pil_image.mode == "P":
            pil_image = pil_image.convert("RGB")
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(str(path), cv_type)

    if image is None:
        print(f"Image {path} is None")
        return None
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
    if image.shape[2] < 3:
        image = np.concatenate((image, image, image), axis=2)
    return image  # (h, w, 3)


def apply_transformations(image, transform):
    from torchvision import transforms

    if transform:
        image = transform(image)
    return image


def save_image(image, path):
    cv2.imwrite(path, image)