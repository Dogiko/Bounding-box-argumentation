# Bounding-box-argumentation

Random crop around box for given image and box position.

## Main function

### random_crop_square(img, start, end, n_crop=1, resize=-1, box_in_crop_ratio=(0.707, 1.414), flip=False)

#### Parameters:

img : array-like, image array (with shape hight x weith x channel)

start : int (y,x), left-up position of bounding box

end : int (y,x), right-down position of bounding box

n_crop : number of output croped image

resize : int, resize each output image to assigned value. Do nothing when value < 0.

box_in_crop_ratio : float $(r, R)$, range of "length of output image / length of box"

flip : bool, random flip(left <-> right) if True.

#### Returns:

List of [crop_img, crop_box_center, crop_half_weith, crop_half_hight]

crop_img : array-like, croped image with same channal as input img

crop_box_center : float (y, x), position of bounding box in croped image(support length 1)

crop_half_weith : float, distence from crop_box_center to left-end or right-end.

crop_half_hight : float, distence from crop_box_center to up-end or down-end.