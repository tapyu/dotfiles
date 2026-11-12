classdef Opset8TranslationStrategy < nnet.internal.cnn.onnx.fcn.matlab.Opset7TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 8.

    %   Copyright 2020-2023 The MathWorks, Inc.
    methods
        nodeTranslation = translateExpand(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMaxPool(~, nodeTranslation, node, IntegerTensorNames);
    end
end
