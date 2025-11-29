from PIL import Image
import numpy as np

# 直接打开mask图片查看
mask_path = "/ssd_group_ssd_data/users/chenjunhui/FLUX_inpainting/data/navsim/seq/test/00ca456004365f38/mask/00_0.png"
mask_img = Image.open(mask_path)

print("图像模式:", mask_img.mode)
print("图像尺寸:", mask_img.size)
print("图像格式:", mask_img.format)

# 转换为numpy数组查看具体数值
mask_array = np.array(mask_img)
print("数组形状:", mask_array.shape)
print("唯一像素值:", np.unique(mask_array))
print("像素值范围:", mask_array.min(), "~", mask_array.max())

# 查看具体像素值
print("前10x10区域的像素值:")
print(mask_array[:10, :10])
