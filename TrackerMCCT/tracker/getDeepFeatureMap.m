function feat = getDeepFeatureMap(im, cos_window, layers)
% Extract deep features from VGG

global net
global enableGPU
if isempty(net)
    initial_net();
end

sz_window = size(cos_window);
if ismatrix(im)
    im = cat(3, im, im, im);
end
% Preprocessing
img = single(im);        % note: [0, 255] range
img = imResample(img, net.normalization.imageSize(1:2));
img = img - net.normalization.averageImage;
if enableGPU, img = gpuArray(img); end
% Run the CNN
res = vl_simplenn(net,img);
% Initialize feature maps
feat = cell(length(layers), 1);

for ii = 1:length(layers)
    % Resize to sz_window
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
    x = imResample(x, sz_window(1:2));
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    feat{ii}=x;
end

end
