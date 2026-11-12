function const = permuteConstInput(const, constDim, inputLayerTypeEnum)
%   Copyright 2018-2019 The MathWorks, Inc.

% Utility function for Add, Sub, Mul, and Div nodes to translate const
% second inputs
constDim = constDim(:)';
if isrow(const)
    % Make sure it has size [C 1]
    const   = const(:);
end
if ~(inputLayerTypeEnum == "RECURRENT" || inputLayerTypeEnum == "FEATURE")
    % Get the onnx dimension and extend it to CHW (2d image input) or
    % CHWD (3d image input)
    if isscalar(constDim)
        constDim = appendOnes(constDim, inputLayerTypeEnum);
    elseif numel(constDim)==2 && constDim(1)==1
        % It's a [1 C] onnx tensor to be added to a [N C] tensor. Make
        % it [C 1 1]
        constDim = appendOnes(constDim(2), inputLayerTypeEnum);
    end
    % Convert from CHW row-major to HWC col-major:
    const = reshape(const, fliplr(constDim));           % Now it's either WHC (2d image input) or DWHC (3d image input) colmaj.
    if inputLayerTypeEnum == "IMAGE3D"
        const = permute(const, [3 2 1 4]);              % Now it's HWDC colmaj
    else
        const = permute(const, [2 1 3]);                % Now it's HWC colmaj.
    end
end
end

function constDim = appendOnes(constDim, inputLayerTypeEnum)
% Utility function to create a 3- elem (2d image inputs) 
% or 4-elem (3d image inputs) dim vector

if inputLayerTypeEnum == "IMAGE3D"
    constDim = [constDim 1 1 1]; 
else
    constDim = [constDim 1 1];
end
end

