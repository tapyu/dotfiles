function validateIONImageInputSize(ImageInputSize)

%   Copyright 2021 The MathWorks, Inc.    

if ~isempty(ImageInputSize) && ~(isreal(ImageInputSize) && isvector(ImageInputSize) && ...
        ismember(numel(ImageInputSize),[3 4]) && ...
        isequal(ImageInputSize, floor(ImageInputSize)) && ...
        all(ImageInputSize > 0))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ImageInputSizeBad')));
end
end