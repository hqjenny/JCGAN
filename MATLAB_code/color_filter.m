function I_filter = color_filter (img, filter)

if (size(img,3)==1)
    fprintf('Input image should be color image\n');
    return;
end

img = im2double(img);

img_reshape = reshape(img, [], 3);
img_filtered = filter * img_reshape';

I_filter = reshape(img_filtered', size(img));

end