import h5py, cv2
import numpy as np


def parse_img_array(data):
    """Decode N encoded images into a uint8 BGR array of shape (N, H, W, C).

    Each input element may be bytes or a uint8 array.
    """
    flat = data.ravel()

    imgs = []
    for buf in flat:
        # Accept either bytes or uint8 arrays.
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported buffer type: {type(buf)}")

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None; the byte stream may not contain a valid image format")
        imgs.append(img)

    return np.stack(imgs, axis=0)


def h5_to_dict(node):
    result = {}
    for name, item in node.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if "rgb" in name:
                result[name] = parse_img_array(data)
            else:
                result[name] = data
        elif isinstance(item, h5py.Group):
            result[name] = h5_to_dict(item)
    # Optional attribute loading:
    if hasattr(node, "attrs") and len(node.attrs) > 0:
        result["_attrs"] = dict(node.attrs)
    return result


def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        data_dict = h5_to_dict(f)
    return data_dict
