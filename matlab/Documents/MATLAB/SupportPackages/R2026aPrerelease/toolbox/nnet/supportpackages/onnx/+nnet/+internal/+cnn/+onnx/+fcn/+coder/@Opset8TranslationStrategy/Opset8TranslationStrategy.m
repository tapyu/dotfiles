classdef Opset8TranslationStrategy < nnet.internal.cnn.onnx.fcn.coder.Opset7TranslationStrategy
    % A strategy class implementing operator translators for opset version
    % 8.

    %   Copyright 2024 The MathWorks, Inc.
    methods
        nodeTranslation = translateExpand(~, nodeTranslation, node, IntegerTensorNames);
        nodeTranslation = translateMaxPool(~, nodeTranslation, node, IntegerTensorNames);
    end
end
