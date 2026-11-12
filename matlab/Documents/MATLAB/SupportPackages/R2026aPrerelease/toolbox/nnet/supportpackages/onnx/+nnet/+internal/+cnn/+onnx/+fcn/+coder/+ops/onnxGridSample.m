function [Y, numDimsY] = onnxGridSample(X_, numDimsX_, grid_, align_corners_, ~, ~)
% Implement ONNX GridSample operator
% align_corners: 0(default), 1
% ONNX X format [N,C,H,W], grid format [N,H_out,W_out,2]
% output format [N,C,H_out,W_out]
% X has reverse-ONNX dimension ordering: [W,H,C,N], grid format [2,W_out,H_out,N]
% output format [W_out,H_out,C,N]
% Compared to MATLAB's convention, the spatial dimensions are transposed in the following code. 
%#codegen

% Copyright 2024-2026 The MathWorks, Inc.

X               = single(nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_));
numDimsX        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
grid            = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(grid_);
align_corners   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(align_corners_);

% Currently supports 'bilinear' mode and 'zeros' padding_mode (asserts will hit in matlab version)

numDimsY = coder.const(numDimsX);
[sizeXW, sizeXH, sizeXC, ~] = size(X);
[~, sizeGridW, sizegridH, ~] = size(grid);
gridH = grid(2,:,:,:);
gridW = grid(1,:,:,:);
gridHUnnorm = iGridSamplerComputeSourceIndex(gridH, sizeXH, align_corners);
gridWUnnorm = iGridSamplerComputeSourceIndex(gridW, sizeXW, align_corners);

% Current only support "bilinear" mode
% grab 4 nearest corner points
% [1,W_out, H_out, N]
gridHUnnorm0 = floor(gridHUnnorm);
gridWUnnorm0 = floor(gridWUnnorm);
gridHUnnorm1 = gridHUnnorm0 + 1;
gridWUnnorm1 = gridWUnnorm0 + 1;

% get pixel value at corner coords over X
% [W_out,H_out,C,N]
Ia = iGetPixelValue(X,gridHUnnorm0,gridWUnnorm0,sizeXW, sizeXH, sizeXC,sizeGridW,sizegridH);
Ib = iGetPixelValue(X,gridHUnnorm0,gridWUnnorm1,sizeXW, sizeXH, sizeXC,sizeGridW,sizegridH);
Ic = iGetPixelValue(X,gridHUnnorm1,gridWUnnorm0,sizeXW, sizeXH, sizeXC,sizeGridW,sizegridH);
Id = iGetPixelValue(X,gridHUnnorm1,gridWUnnorm1,sizeXW, sizeXH, sizeXC,sizeGridW,sizegridH);

% get distance between center points and corners
% [W_out,H_out,1,N]
Wd = permute((gridHUnnorm - gridHUnnorm0) .* (gridWUnnorm- gridWUnnorm0),[2,3,1,4]);
Wc = permute((gridHUnnorm - gridHUnnorm0) .* (gridWUnnorm1- gridWUnnorm),[2,3,1,4]);
Wb = permute((gridHUnnorm1 - gridHUnnorm) .* (gridWUnnorm- gridWUnnorm0),[2,3,1,4]);
Wa = permute((gridHUnnorm1 - gridHUnnorm) .* (gridWUnnorm1- gridWUnnorm),[2,3,1,4]);

% sum the interpolation values together
% [W_out,H_out,C,N]
Y = Ia.*Wa + Ib.*Wb + Ic.*Wc + Id.*Wd;

% sub-functions
    function coord = iGridSamplerComputeSourceIndex(coord, size, align_corners)
        % unnormalized grid coordination
        coord = iGridSamplerUnnormalize(coord, size, align_corners);
        % add one on coord for 1-index
        coord = coord+1;
    end

    function coord = iGridSamplerUnnormalize(coord, size, align_corners)
        if align_corners
            % unnormalize coord from [-1, 1] to [0, size - 1]
            coord = ((coord + 1) / 2) * (size - 1);
        else
            % unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
            coord = ((coord + 1) * size - 1) / 2;
        end
    end

    function pixelValue = iGetPixelValue(X,gridH,gridW,sizeXW, sizeXH, sizeXC,sizeGridW,sizeGridH)
        % gridH, gridW in [1,W_out,H_out,N]
        % pixelValue in [Wout,H_out,C,N]
        % X in [W_in,H_in,C,N]

        % for each N, go over each element in gridH and gridW [1,W_out,H_out]
        pixelValue = zeros(sizeGridW, sizeGridH, sizeXC, size(gridH,4));
        for i= 1:size(X,4)
            tempGridH = reshape(gridH(:,:,:,i),1,[]);
            tempGridW = reshape(gridW(:,:,:,i),1,[]);
            tmpPixelValue = zeros(sizeGridW*sizeGridH, sizeXC);
            for j = 1:size(tempGridH,2)
                % when the padding_mode is set to "zeros", out of bound value is
                % set to 0. The pixelValue for that pixel is 0. At this
                % point, use "zeros" as default
                if tempGridH(j)<1 || tempGridH(j)>sizeXH || tempGridW(j)<1 || tempGridW(j)>sizeXW
                    tmpX = zeros(1, 1, sizeXC, 'single');
                else
                    tmpX = X(tempGridW(j),tempGridH(j),:,i);
                end
                tmpPixelValue(j,:) = tmpX;
            end
            pixelValue(:,:,:,i) = reshape(tmpPixelValue,[sizeGridW sizeGridH sizeXC]);
        end
    end
end