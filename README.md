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



### random_crop_square(img, start, end, n_crop=1, resize=-1, avoid_ratio=(0.707, 1.414), flip=False, minimum_length=32, max_try=100)

#### Parameters:

img : array-like, image array (with shape hight x weith x channel)

starts : int numpy array nx2, left-up position of each bounding box, n=number of bounding box

ends : int numpy_array nx2, right-down position of each bounding box, n=number of bounding box

n_crop : number of output croped image

resize : int, resize each output image to assigned value. Do nothing when value < 0.

avoid_ratio : float $(r, R)$, to exclude result which contained some box center and "length of  box / length of square" in this range.

flip : bool, random flip(left <-> right) if True.

minimum_length : int>=0, minimun length(pixel) for area on image.

max_try : int>0, maxmun times of retry if each random result is excluded(see avoid_ratio above).

#### Returns:

List of crop_img

crop_img : array-like, croped image with same channal as input img

### img_to_torch(numpy_img)

convert numpy array(0~255) to torch tensor in float32(-1.~+1.), 

#### Parameters:

numpy_img: uint8 numpy array with shape (n_datum, side_x, side_y, n_channel)

#### Returns:

torch_input : float torch tensor with dim (n_datum, n_channel, side_x, side_y)