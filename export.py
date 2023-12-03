import paddle
import vision_transformer as vits
from vision_transformer import DINOHead
from utils import MultiCropWrapper

if __name__ == "__main__":
    teacher = vits.__dict__["vit_small"](patch_size=16)
    model = MultiCropWrapper(teacher, DINOHead(384, 65536))
    x = paddle.randn(shape=[1, 3, 224, 768])
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(model, input_spec=(x,), path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
