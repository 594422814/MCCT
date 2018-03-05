function out = getFeatureMap(im_patch, cf_response_size, hog_cell_size)
% Extract HOG features

temp = fhog(single(im_patch), hog_cell_size);
h = cf_response_size(1);
w = cf_response_size(2);
out = zeros(h, w, 28, 'single');
out(:,:,2:28) = temp(:,:,1:27);
if hog_cell_size > 1
    im_patch = mexResize(im_patch, [h, w] ,'auto');
end
% if color image
if size(im_patch, 3) > 1
    im_patch_gray = rgb2gray(im_patch);
else
    im_patch_gray = im_patch;
end
out(:,:,1) = single(im_patch_gray)/255 - 0.5; 

end

